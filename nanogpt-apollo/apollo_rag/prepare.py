"""Build a complete private corpus, then publish it by a same-volume rename."""

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

from . import __version__
from .chunks import document_chunks
from .inputs import EXTENSIONS, SKIP_DIRS, collect_inputs, linked, sha256
from .provenance import file_svn, read_svn
from .tokenizer import DEFAULT_CACHE, load_tokenizer


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def validate_paths(manuals, source_root, output):
    if not manuals and source_root is None:
        raise ValueError("Supply at least one --manual or a --source-root")
    manuals = [Path(value).absolute() for value in manuals]
    selected = [*manuals, Path(output).absolute()]
    if source_root is not None:
        selected.append(Path(source_root).absolute())
    # Reject UNC targets before stat/read operations, including on Windows.
    if any(str(path).startswith(("\\\\", "//")) for path in selected):
        raise ValueError("Use local files and directories; network share paths are not supported")
    if len({path.name.casefold() for path in manuals}) != len(manuals):
        raise ValueError("Manual names must be unique so their relative paths remain unambiguous")
    for path in manuals:
        if not path.is_file() or linked(path):
            raise ValueError("Every --manual must be an existing local regular file")
    source_root = Path(source_root).absolute() if source_root is not None else None
    if source_root is not None and (not source_root.is_dir() or linked(source_root)):
        raise ValueError("--source-root must be a regular local directory")
    # Phase 1 does not access network shares, directly or through mapped drives.
    inputs = manuals + ([source_root] if source_root is not None else [])
    for path in [*inputs, Path(output).absolute()]:
        if str(path.resolve()).startswith(("\\\\", "//")):
            raise ValueError("Use local files and directories; network share paths are not supported")
    output = Path(output).resolve()
    manuals = [path.resolve() for path in manuals]
    source_root = source_root.resolve() if source_root is not None else None
    if source_root is not None and (output == source_root or source_root in output.parents):
        raise ValueError("--output must be outside --source-root to prevent recursive ingestion")
    if output in manuals or any(output in manual.parents for manual in manuals):
        raise ValueError("--output cannot contain or replace an input manual")
    if output.exists():
        raise ValueError("--output already exists. Choose a new directory; existing bundles are never overwritten")
    return manuals, source_root, output


def prepare(manuals, source_root, output, *, tokenizer=None, tokenizer_cache=DEFAULT_CACHE,
            tokenizer_revision="main", offline=False, max_tokens=384, overlap_tokens=32,
            max_file_bytes=2 * 1024 * 1024, max_total_bytes=64 * 1024 * 1024):
    manuals, source_root, output = validate_paths(manuals, source_root, output)
    if not 32 <= max_tokens <= 384:
        raise ValueError("--max-tokens must be between 32 and 384, including prefix and special tokens")
    if not 0 <= overlap_tokens <= max_tokens // 4:
        raise ValueError("--overlap-tokens must be nonnegative and at most one quarter of --max-tokens")
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("Input byte limits must be positive")
    cache = Path(tokenizer_cache).resolve()
    if str(cache).startswith(("\\\\", "//")):
        raise ValueError("--tokenizer-cache must be a local directory")
    if source_root is not None and (cache == source_root or source_root in cache.parents):
        raise ValueError("--tokenizer-cache must be outside the read-only source checkout")
    # Input validation and filtering occur before creating output artifacts.
    documents, exclusions = collect_inputs(manuals, source_root, max_file_bytes, max_total_bytes)
    if not documents:
        raise ValueError("No safe readable input documents remain; check input selection and limits")
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_cache, tokenizer_revision, offline)
    svn_files, svn_summary = read_svn(source_root) if source_root else (
        {}, {"availability": "not-applicable", "reason": "manuals-only"}
    )
    all_chunks, sources = [], []
    for document in documents:
        svn = file_svn(document.path, svn_files, svn_summary) if document.kind == "source" else {
            "availability": "not-applicable", "revision": None, "local_status": None,
        }
        chunks, cleanup = document_chunks(document, tokenizer, svn, max_tokens, overlap_tokens)
        all_chunks.extend(chunks)
        sources.append({
            "kind": document.kind, "path": document.path, "absolute_path": document.absolute_path,
            "source_hash": document.source_hash, "encoding": document.encoding,
            "svn": svn, "chunks": len(chunks), "cleanup": cleanup,
        })
    if not all_chunks:
        raise ValueError("Inputs contain no passages after cleanup")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".apollo-prepare-", dir=output.parent))
    try:
        # Even a custom output folder inside the application stays private in Git.
        (staging / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
        write_jsonl(staging / "chunks.jsonl", all_chunks)
        write_jsonl(staging / "exclusions.jsonl", exclusions)
        sample = ["# Private sample passages", "", "This file contains selected private corpus content.", ""]
        for kind in ("manual", "source"):
            for chunk in [item for item in all_chunks if item["kind"] == kind][:3]:
                sample += [f"## {chunk['path']}:{chunk['line_start']}-{chunk['line_end']}",
                           "", chunk["text"], ""]
        (staging / "sample_passages.md").write_text("\n".join(sample), encoding="utf-8", newline="\n")
        hashes = {
            name: sha256((staging / name).read_bytes())
            for name in ("chunks.jsonl", "exclusions.jsonl", "sample_passages.md")
        }
        manifest = {
            "schema_version": 1, "application_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manuals": [str(path) for path in manuals],
            "source_root": str(source_root) if source_root else None,
            "svn": svn_summary, "tokenizer": tokenizer.identity,
            "settings": {"max_tokens": max_tokens, "overlap_tokens": overlap_tokens,
                         "max_file_bytes": max_file_bytes, "max_total_bytes": max_total_bytes,
                         "source_encodings": ["utf-8-sig", "cp1250"], "manual_encoding": "utf-8-sig",
                         "extensions": sorted(EXTENSIONS), "excluded_directories": sorted(SKIP_DIRS)},
            "sources": sources, "document_count": len(sources), "chunk_count": len(all_chunks),
            "exclusion_counts": dict(sorted(Counter(item["reason"] for item in exclusions).items())),
            "files": hashes,
        }
        # Timestamp is informational; corpus fingerprint is stable across reruns.
        stable = {key: value for key, value in manifest.items() if key != "created_at"}
        manifest["corpus_fingerprint"] = sha256(json_bytes(stable))
        (staging / "manifest.json").write_bytes(json_bytes(manifest))
        # os.rename fails if the target appeared concurrently; never replace it.
        if output.exists():
            raise ValueError("--output appeared during preparation; choose a new directory")
        os.rename(staging, output)
    finally:
        if staging.exists():
            # Only the exact staging directory created above is ever removed.
            if staging.resolve().parent != output.parent or not staging.name.startswith(".apollo-prepare-"):
                raise RuntimeError("Staging cleanup refused an unexpected directory")
            shutil.rmtree(staging)
    return manifest
