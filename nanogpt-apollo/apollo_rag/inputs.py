"""Conservative source selection and content screening before corpus output."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat

EXTENSIONS = frozenset({
    ".sql", ".pks", ".pkb", ".pls", ".prc", ".fnc", ".trg", ".py",
    ".js", ".ts", ".html", ".htm", ".css", ".less", ".xml", ".json", ".md", ".txt",
})
SKIP_DIRS = frozenset({
    ".svn", ".git", ".codegraph", ".cache", ".codex", ".agents", ".aws", ".cursor",
    "awesome-codex-subagents",
    "node_modules", "vendor", "webix", "webix_pro", "webix_debug", "bower_components",
    "codebase_7_2_7",
    "venv", ".venv", "__pycache__", "dist", "build", "coverage", "logs",
    "evidence", "evidence_dumps", "screenshots", "models", "tokenizers", "indexes",
})
VERSIONED_WEBIX_DIR = re.compile(r"webix[-_]\d+(?:[._-]\d+)*(?:[-_](?:pro|debug))?", re.I)
CONFIG_NAME = re.compile(
    r"(^|[._-])(config(?:uration)?|settings|credentials?|secrets?|connections?|"
    r"dbconfig|database|passwords?|tokens?)([._-]|$)", re.I,
)
GENERATED_NAME = re.compile(
    r"(^|[._-])(inventory|dataset|dump|evidence|snapshot|report|har|trace)([._-]|$)", re.I,
)
SECRET_ASSIGNMENT = re.compile(
    r"[\"']?(?:password|passwd|pwd|secret|client_secret|api[_-]?key|access[_-]?token|"
    r"auth[_-]?token|authorization|connection[_-]?string)[\"']?\s*[:=]\s*"
    r"(?P<value>[^\r\n,;}]+)", re.I,
)
XML_SECRET = re.compile(
    r"<(?P<tag>(?:[\w.-]+:)?(?:password|passwd|pwd|secret|client[_-]?secret|"
    r"api[_-]?key|access[_-]?token|auth[_-]?token|authorization|connection[_-]?string))"
    r"\b[^>]*>(?P<value>.*?)</(?P=tag)\s*>", re.I | re.S,
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b(?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|oracle)://[^\s/:]+:[^\s/@]+@", re.I),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9_\-.]{16,}", re.I),
    re.compile(
        r"\b(?:(?:create|alter)\s+user\s+|"
        r"(?:create|alter)\s+(?:public\s+)?database\s+link\s+(?:\"[^\"]+\"|[\w$#.]+)\s+connect\s+to\s+)"
        r"(?:\"[^\"]+\"|[\w$#]+)\s+identified\s+by\s+[\"']?[^\s;\"']{3,}", re.I,
    ),
    re.compile(r"\bconnect\s+[^\s/]+/[^\s@]+@[^\s;]+", re.I),
)


@dataclass(frozen=True)
class SourceDocument:
    kind: str
    path: str
    absolute_path: str
    source_hash: str
    encoding: str
    text: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def linked(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def secret_bearing(text: str) -> bool:
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return True
    for match in XML_SECRET.finditer(text):
        value = match.group("value").strip()
        # XML markup (including CDATA) is content, not an angle-bracket
        # placeholder. Only complete environment references and this explicit
        # implementation marker are safe nonempty XML credential values.
        if value and not re.fullmatch(
            r"(?:\$\{[A-Z_][A-Z0-9_]*\}|\$[A-Z_][A-Z0-9_]*|"
            r"TODO: load from env or secrets vault)", value, re.I,
        ):
            return True
    values = [match.group("value") for match in SECRET_ASSIGNMENT.finditer(text)]
    for value in values:
        value = value.strip()
        # Empty literals, environment lookups, bind parameters and explicit
        # placeholders do not contain an actual credential.
        cleaned = value.strip("\"' \t")
        if not cleaned:
            continue
        if re.match(r"^(?:<[^>]+>|\$\{|\$[A-Z_]|:[A-Za-z_]|\?|None\b|null\b|"
                    r"os\.(?:environ|getenv)|process\.env|(?:getenv|env)\s*\(|"
                    r"(?:your[_ -]|example|placeholder|changeme|redacted|TODO))", cleaned, re.I):
            continue
        # Treat quoted literals and unquoted credential-like assignments as unsafe.
        return True
    return False


def exclusion(path, reason, **extra):
    return {"path": str(path), "reason": reason, **extra}


def source_reason(path: Path):
    name = path.name.lower()
    if name.startswith(".env") or CONFIG_NAME.search(name):
        return "configuration-or-credentials"
    if name in {"package-lock.json", "package.json", "yarn.lock", "uv.lock", "pnpm-lock.yaml"}:
        return "dependency-metadata"
    if ".min." in name or name.endswith(".map") or name.startswith("webix"):
        return "vendor-or-minified"
    if path.suffix.lower() not in EXTENSIONS:
        return "unsupported-extension"
    if GENERATED_NAME.search(name):
        return "generated-evidence-or-inventory"
    # Forms SQL under analysis/** is valuable implementation evidence. Only
    # generated JSON inventories in these directories are rejected by location.
    if "analysis" in [part.lower() for part in path.parts] and path.suffix.lower() == ".json":
        return "generated-analysis-json"
    return None


def read_document(path: Path, kind: str, relative: str, max_bytes: int):
    if linked(path):
        return None, exclusion(relative, "symlink-or-junction", kind=kind)
    if path.stat().st_size > max_bytes:
        return None, exclusion(relative, "file-size-limit", kind=kind)
    raw = path.read_bytes()
    if b"\x00" in raw:
        return None, exclusion(relative, "binary-content", kind=kind)
    text = None
    used_encoding = None
    # Manuals are explicitly UTF-8. Legacy source can contain CP1250 comments.
    for encoding in (("utf-8-sig",) if kind == "manual" else ("utf-8-sig", "cp1250")):
        try:
            text = raw.decode(encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return None, exclusion(relative, "unsupported-text-encoding", kind=kind)
    if any(ord(char) < 32 and char not in "\t\r\n\f" for char in text):
        return None, exclusion(relative, "binary-control-characters", kind=kind)
    if secret_bearing(text):
        return None, exclusion(relative, "credential-bearing-content", kind=kind)
    if not text.strip():
        return None, exclusion(relative, "empty-content", kind=kind)
    return SourceDocument(kind, relative, str(path.resolve()), sha256(raw), used_encoding, text), None


def collect_inputs(manuals, source_root, max_file_bytes=2 * 1024 * 1024,
                   max_total_bytes=64 * 1024 * 1024):
    documents, exclusions = [], []
    total = 0

    def take(path, kind, relative):
        nonlocal total
        document, rejected = read_document(path, kind, relative, max_file_bytes)
        if rejected:
            exclusions.append(rejected)
        else:
            total += path.stat().st_size
            if total > max_total_bytes:
                raise ValueError("Selected readable inputs exceed --max-total-bytes; increase the explicit limit")
            documents.append(document)

    for manual in sorted(manuals, key=lambda path: path.name):
        take(manual, "manual", manual.name)
    if source_root is not None:
        def walk_error(_error):
            raise ValueError("A source directory could not be read; check local permissions") from None

        for current, dirs, files in os.walk(source_root, followlinks=False, onerror=walk_error):
            current_path = Path(current)
            kept = []
            for dirname in sorted(dirs):
                child = current_path / dirname
                relative = child.relative_to(source_root).as_posix()
                if linked(child):
                    exclusions.append(exclusion(relative, "symlink-or-junction", kind="source"))
                elif dirname.lower() in SKIP_DIRS or VERSIONED_WEBIX_DIR.fullmatch(dirname):
                    exclusions.append(exclusion(relative, "excluded-directory", kind="source"))
                else:
                    kept.append(dirname)
            dirs[:] = kept
            for filename in sorted(files):
                child = current_path / filename
                relative = child.relative_to(source_root)
                reason = source_reason(relative)
                if reason:
                    exclusions.append(exclusion(relative.as_posix(), reason, kind="source"))
                else:
                    take(child, "source", relative.as_posix())
    return documents, exclusions


@dataclass(frozen=True)
class Line:
    text: str
    number: int
    page: int | None
    column_start: int = 0


def original_lines(document: SourceDocument):
    """Physical source lines; form feed changes page without changing line count."""
    result = []
    page = 1
    for number, physical in enumerate(document.text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        fragments = physical.split("\f") if document.kind == "manual" else [physical]
        column_start = 0
        for index, fragment in enumerate(fragments):
            if index:
                page += 1
            result.append(Line(fragment, number, page if document.kind == "manual" else None, column_start))
            column_start += len(fragment) + 1
    return result


def clean_manual(lines):
    pages = {}
    for line in lines:
        if line.text.strip():
            pages.setdefault(line.page, []).append(line)
    candidates = Counter()
    for page_lines in pages.values():
        candidates.update({line.text.strip() for line in page_lines[:2] + page_lines[-2:]})
    repeated = {
        value for value, count in candidates.items()
        if len(pages) >= 3 and count >= max(3, (len(pages) * 3 + 4) // 5)
        and not value.isdecimal()
    }
    cleaned, removed = [], []
    for line in lines:
        text = line.text.strip()
        page_lines = pages.get(line.page, [])
        at_edge = line in page_lines[:2] + page_lines[-2:]
        reason = None
        page_number = re.fullmatch(r"(?:(page)\s+)?(\d+)(?:\s+(of|/)\s+\d+)?", text, re.I)
        if text in repeated and at_edge:
            reason = "repeated-header-footer"
        elif page_number and at_edge and (
            page_number.group(1) or page_number.group(3)
            or (len(pages) > 1 and line in page_lines[-2:]
                and int(page_number.group(2)) == line.page)
        ):
            reason = "standalone-page-number"
        elif re.search(r"\.{3,}\s*\d+\s*$", text):
            reason = "table-of-contents-entry"
        elif text.casefold() in {"table of contents", "contents"}:
            reason = "table-of-contents-heading"
        if reason:
            removed.append({"line_start": line.number, "line_end": line.number,
                            "page": line.page, "reason": reason})
            # Removed lines still separate paragraphs.
            cleaned.append(Line("", line.number, line.page, line.column_start))
        else:
            cleaned.append(line)
    return cleaned, removed
