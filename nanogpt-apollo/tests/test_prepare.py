import json
from pathlib import Path

import pytest

from apollo_rag.cli import main
from apollo_rag.prepare import prepare


def manual(tmp_path, text="Stock accounts\n\nChoose the stock account before posting."):
    path = tmp_path / "manual.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_manuals_only_bundle_schema_fingerprint_and_deterministic_content(tmp_path, tokenizer):
    path = manual(tmp_path)
    first = prepare([path], None, tmp_path / "one", tokenizer=tokenizer)
    second = prepare([path], None, tmp_path / "two", tokenizer=tokenizer)
    assert first["svn"]["availability"] == "not-applicable"
    assert first["corpus_fingerprint"] == second["corpus_fingerprint"]
    assert first["files"] == second["files"]
    assert first["sources"][0]["absolute_path"] == str(path.resolve())
    assert first["tokenizer"]["revision"] == "fixture-v1"
    chunks = [json.loads(line) for line in (tmp_path / "one/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert chunks and all(chunk["kind"] == "manual" for chunk in chunks)
    assert first["sources"][0]["source_hash"] == chunks[0]["source_hash"]
    assert (tmp_path / "one/.gitignore").read_text() == "*\n!.gitignore\n"


def test_synthetic_secret_never_echoed_into_any_artifact(tmp_path, tokenizer):
    root = tmp_path / "source"
    root.mkdir()
    literal = "private-fixture-literal-974"
    cdata_literal = "fixture-private-value-974"
    (root / "unsafe.py").write_text(f"password = '{literal}'", encoding="utf-8")
    (root / "metadata.xml").write_text(f"<connection><password>{literal}</password></connection>", encoding="utf-8")
    (root / "document.xml").write_text(
        f"<connection><password><![CDATA[{cdata_literal}]]></password></connection>", encoding="utf-8",
    )
    (root / "safe.sql").write_text("-- stock account posting\nbegin post_stock; end;", encoding="utf-8")
    output = tmp_path / "bundle"
    result = prepare([manual(tmp_path)], root, output, tokenizer=tokenizer)
    exclusions = [json.loads(line) for line in (output / "exclusions.jsonl").read_text().splitlines()]
    assert {"kind": "source", "path": "unsafe.py", "reason": "credential-bearing-content"} in exclusions
    assert {"kind": "source", "path": "metadata.xml", "reason": "credential-bearing-content"} in exclusions
    assert {"kind": "source", "path": "document.xml", "reason": "credential-bearing-content"} in exclusions
    for path in output.iterdir():
        assert literal not in path.read_text(encoding="utf-8")
        assert cdata_literal not in path.read_text(encoding="utf-8")
    assert [source["path"] for source in result["sources"]] == ["manual.txt", "safe.sql"]
    assert result["sources"][1]["svn"]["local_status"] == "unknown"


def test_identified_by_manual_prose_is_retained_in_bundle(tmp_path, tokenizer):
    path = manual(tmp_path, "Locations are identified by their code.\n\nChoose a stock account.")
    output = tmp_path / "bundle"
    result = prepare([path], None, output, tokenizer=tokenizer)
    assert result["document_count"] == 1
    assert "Locations are identified by their code." in (output / "chunks.jsonl").read_text(encoding="utf-8")
    assert not (output / "exclusions.jsonl").read_text(encoding="utf-8")


def test_existing_output_is_preserved_on_error(tmp_path, tokenizer):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user.txt"
    marker.write_text("user content", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        prepare([manual(tmp_path)], None, output, tokenizer=tokenizer)
    assert marker.read_text() == "user content"
    assert [path.name for path in output.iterdir()] == ["user.txt"]


def test_partial_write_failure_does_not_publish_bundle_or_leave_staging(tmp_path, tokenizer, monkeypatch):
    path = manual(tmp_path)
    output = tmp_path / "bundle"
    original = Path.write_bytes

    def fail_manifest(self, data):
        if self.name == "manifest.json":
            raise OSError("simulated disk full")
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", fail_manifest)
    with pytest.raises(OSError):
        prepare([path], None, output, tokenizer=tokenizer)
    assert not output.exists()
    assert not list(tmp_path.glob(".apollo-prepare-*"))


def test_output_inside_source_is_rejected_before_loading_tokenizer(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(ValueError, match="outside --source-root"):
        prepare([], root, root / "bundle")
    assert not (root / "bundle").exists()


def test_cache_cannot_modify_source_checkout(tmp_path, tokenizer):
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(ValueError, match="read-only source"):
        prepare([], root, tmp_path / "bundle", tokenizer=tokenizer, tokenizer_cache=root / "tokenizer")


def test_failed_tokenizer_loading_has_no_output(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("E5 tokenizer unavailable")

    monkeypatch.setattr("apollo_rag.prepare.load_tokenizer", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        prepare([manual(tmp_path)], None, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("kwargs", [
    {"max_tokens": 385}, {"max_tokens": 31}, {"overlap_tokens": 100},
    {"max_file_bytes": 0}, {"max_total_bytes": -1},
])
def test_invalid_limits_are_explicit(tmp_path, tokenizer, kwargs):
    with pytest.raises(ValueError):
        prepare([manual(tmp_path)], None, tmp_path / "bundle", tokenizer=tokenizer, **kwargs)


def test_cli_returns_failure_without_dumping_os_exception(tmp_path, monkeypatch, capsys):
    def denied(**kwargs):
        raise PermissionError("sensitive proxy value")

    monkeypatch.setattr("apollo_rag.cli.prepare", denied)
    assert main(["prepare", "--manual", str(tmp_path / "missing.txt"), "--output", str(tmp_path / "bundle")]) == 1
    assert "sensitive proxy value" not in capsys.readouterr().err


def test_actual_cli_prepares_explicit_manual_with_injected_tokenizer(tmp_path, tokenizer, monkeypatch, capsys):
    path = manual(tmp_path)
    output = tmp_path / "from-cli"
    monkeypatch.setattr("apollo_rag.prepare.load_tokenizer", lambda *args, **kwargs: tokenizer)
    assert main(["prepare", "--manual", str(path), "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manuals"] == [str(path.resolve())]
    assert manifest["document_count"] == 1
    assert "Prepared 1 documents" in capsys.readouterr().out
