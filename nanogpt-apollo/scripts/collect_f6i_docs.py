"""Inventory a Windows document share and make a local PDF/text copy."""

import argparse
import csv
import filecmp
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


FORMATS = {".pdf", ".docx", ".doc"}
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def is_link(path):
    return path.is_symlink() or path.is_junction()


def inventory(source, scan_errors):
    for folder, directories, files in os.walk(source, onerror=scan_errors.append):
        root = Path(folder)
        directories[:] = sorted(name for name in directories if not is_link(root / name))
        for name in sorted(files):
            path = root / name
            if path.suffix.lower() in FORMATS and not is_link(path):
                yield path


def local_output(root, relative):
    target = root / relative
    if not target.resolve().is_relative_to(root):
        raise ValueError("Output path points outside its destination folder")
    return target


def copy_pdf(original, target):
    if target.exists():
        return "already-present" if filecmp.cmp(original, target, shallow=False) else "conflict"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(original, temporary)
        # Windows rename fails if another file already exists at the destination.
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)
    return "copied"


def pdf_text(path, command, wsl):
    filename = str(path)
    if wsl:
        filename = subprocess.check_output(
            [*command, "wslpath", "-a", filename], timeout=30
        ).decode("utf-8").strip()
    return subprocess.check_output(
        [*command, "pdftotext", "-layout", "-enc", "UTF-8", filename, "-"],
        stderr=subprocess.PIPE,
        timeout=180,
    ).decode("utf-8")


def docx_text(path):
    with ZipFile(path) as document:
        with document.open("word/document.xml") as main:
            root = ElementTree.parse(main).getroot()
    inline = {WORD_NS + "tab": "\t", WORD_NS + "br": "\n", WORD_NS + "cr": "\n"}
    paragraphs = []
    for paragraph in root.iter(WORD_NS + "p"):
        text = "".join(
            node.text or "" if node.tag == WORD_NS + "t" else inline.get(node.tag, "")
            for node in paragraph.iter()
        )
        if text.strip():
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def process(path, relative, pdf_root, text_root, command, wsl, row):
    extension = path.suffix.lower()
    if extension in {".doc", ".docx"} and path.name.startswith("~$"):
        row["text"] = "word-lock-file"
        return
    if extension == ".doc":
        row["text"] = "convert-doc-manually"
        return

    if extension == ".pdf":
        copied = local_output(pdf_root, relative)
        row["pdf_copy"] = copy_pdf(path, copied)
        if row["pdf_copy"] == "conflict":
            row["text"] = "skipped-copy-conflict"
            return
        if command is None:
            row["text"] = "pdftotext-unavailable"
            return
        content = pdf_text(copied, command, wsl)
    else:
        content = docx_text(path)

    if not content.strip():
        row["text"] = "no-text-ocr-needed" if extension == ".pdf" else "empty-docx"
        return
    target = local_output(text_root, relative.with_name(relative.name + ".txt"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    row["text"] = "extracted"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"V:\F6i\doc"))
    parser.add_argument("--pdf-out", type=Path, default=Path(r"D:\F6i\doc"))
    parser.add_argument("--text-out", type=Path, default=Path(r"D:\F6i\doc-text"))
    parser.add_argument("--wsl", metavar="DISTRO", help="Use pdftotext in this WSL distribution")
    parser.add_argument("--apply", action="store_true", help="Copy PDFs and extract text")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Run this script with Windows Python so it can see the Windows drives")
    source, pdf_root, text_root = (path.resolve() for path in (args.source, args.pdf_out, args.text_out))
    if not source.is_dir():
        parser.error(f"Source folder is unavailable to this Windows session: {args.source}")
    roots = (source, pdf_root, text_root)
    if any(a.is_relative_to(b) for a in roots for b in roots if a is not b):
        parser.error("Source, PDF output, and text output must be separate folders")

    scan_errors = []
    paths = list(inventory(source, scan_errors))
    if scan_errors:
        for error in scan_errors:
            print(f"Unable to scan: {error}")
        parser.error("Inventory is incomplete. Check the folder permissions and retry.")
    file_counts = Counter(path.suffix.lower() for path in paths)
    for extension in sorted(FORMATS):
        size = sum(path.stat().st_size for path in paths if path.suffix.lower() == extension)
        print(f"{extension}: {file_counts[extension]} files, {size / (1024 ** 2):.1f} MiB")
    if not args.apply:
        print("Preview only. Use --apply to copy PDFs and extract text.")
        return

    command = ["wsl.exe", "-d", args.wsl, "--exec"] if args.wsl else []
    try:
        subprocess.run([*command, "pdftotext", "-v"], capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        print("pdftotext is unavailable. PDFs will be copied and listed for later extraction.")
        command = None

    report = local_output(text_root, Path("inventory.csv"))
    report.parent.mkdir(parents=True, exist_ok=True)
    outcomes = Counter()
    with report.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("source", "format", "pdf_copy", "text", "error")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number, path in enumerate(paths, 1):
            relative = path.relative_to(source)
            row = dict(source=str(relative), format=path.suffix.lower(), pdf_copy="", text="", error="")
            try:
                process(path, relative, pdf_root, text_root, command, args.wsl, row)
            except (OSError, ValueError, subprocess.SubprocessError, BadZipFile,
                    KeyError, ElementTree.ParseError) as error:
                row["text"] = "error"
                row["error"] = type(error).__name__
            writer.writerow(row)
            outcomes[row["text"]] += 1
            if number % 25 == 0:
                handle.flush()
                print(f"Processed {number}/{len(paths)} files", flush=True)
    print(f"Report: {report}")
    for status, count in sorted(outcomes.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
