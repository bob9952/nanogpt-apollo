"""Verify portable corpus artifacts without opening the original input paths."""

import hashlib
import json
from pathlib import Path

from .chunks import embedding_text
from .inputs import linked, sha256
from .prepare import json_bytes

ARTIFACTS = {"chunks.jsonl", "exclusions.jsonl", "sample_passages.md"}


def local_path(value):
    path = Path(value).absolute()
    if str(path).startswith(("\\\\", "//")) or str(path.resolve()).startswith(("\\\\", "//")):
        raise ValueError("Use local corpus and index paths; network shares are not supported")
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or (candidate.exists() and linked(candidate)):
            raise ValueError("Corpus and index paths must not be symlinks or junctions")
    return path.resolve()


def load_bundle(directory):
    directory = local_path(directory)
    try:
        manifest_path = directory / "manifest.json"
        if linked(manifest_path):
            raise ValueError
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError
        stable = {key: value for key, value in manifest.items() if key not in {"created_at", "corpus_fingerprint"}}
        if manifest.get("corpus_fingerprint") != sha256(json_bytes(stable)):
            raise ValueError
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != ARTIFACTS:
            raise ValueError
        for name, expected in files.items():
            path = directory / name
            if linked(path):
                raise ValueError
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError
        chunks, ids = [], set()
        with (directory / "chunks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    raise ValueError
                identity = [chunk["kind"], chunk["path"], chunk["source_hash"], chunk["paragraph_index"],
                            chunk["passage_start"], chunk["passage_end"], chunk["embedding_text"]]
                expected_id = sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8"))
                if (chunk["id"] != expected_id or chunk["id"] in ids
                        or chunk["kind"] not in {"manual", "source"}
                        or not isinstance(chunk["path"], str) or not chunk["path"]
                        or not isinstance(chunk["text"], str) or not chunk["text"]
                        or chunk["sha256"] != sha256(chunk["text"].encode("utf-8"))
                        or chunk["embedding_text"] != embedding_text(chunk["text"], chunk["embedding_section"])
                        or not 1 <= chunk["line_start"] <= chunk["line_end"]
                        or (chunk["kind"] == "manual"
                            and not 1 <= chunk["page_start"] <= chunk["page_end"])
                        or (chunk["kind"] == "source"
                            and (chunk["page_start"] is not None or chunk["page_end"] is not None))):
                    raise ValueError
                ids.add(chunk["id"])
                chunks.append(chunk)
        if not chunks or len(chunks) != manifest.get("chunk_count"):
            raise ValueError
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise ValueError("Corpus bundle is missing, corrupt, or incompatible; prepare or copy a complete schema-1 bundle") from None
    return manifest, chunks
