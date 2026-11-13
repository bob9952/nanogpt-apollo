from html import escape
import subprocess

from apollo_rag.provenance import file_svn, parse_svn, read_svn


def test_per_file_revisions_and_dirty_properties_are_preserved(tmp_path):
    paths = [escape(str(tmp_path / name), quote=True) for name in ("stock.sql", "ledger.sql")]
    info = f'<info><entry path="{paths[0]}" revision="177"><commit revision="170"/><url>https://example.invalid/svn/stock.sql</url></entry><entry path="{paths[1]}" revision="178"><commit revision="178"/></entry></info>'
    status = f'<status><target path="{escape(str(tmp_path), quote=True)}"><entry path="{paths[0]}"><wc-status item="modified" props="modified" revision="177"/></entry></target></status>'
    files, summary = parse_svn(info, status, tmp_path)
    assert files["stock.sql"]["revision"] == "177"
    assert files["stock.sql"]["last_changed_revision"] == "170"
    assert files["stock.sql"]["local_status"] == "modified"
    assert files["stock.sql"]["property_status"] == "modified"
    assert files["ledger.sql"]["revision"] == "178"
    assert files["ledger.sql"]["local_status"] == "normal"
    assert file_svn("new.sql", files, summary)["local_status"] == "unversioned"


def test_no_svn_metadata_means_unavailable_never_assumed_pristine(tmp_path):
    files, summary = read_svn(tmp_path)
    provenance = file_svn("file.sql", files, summary)
    assert summary["reason"] == "no-working-copy-metadata"
    assert provenance["availability"] == "unavailable"
    assert provenance["revision"] is None
    assert provenance["local_status"] == "unknown"


def test_local_svn_queries_never_use_url_or_update_flags(tmp_path, monkeypatch):
    (tmp_path / ".svn").mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert command[-1] == "."
        assert kwargs["cwd"] == tmp_path
        assert "-u" not in command and "--show-updates" not in command
        assert not any(argument.startswith("https:") for argument in command)
        info = '<info><entry path="." revision="178"><url>https://example.invalid/svn/app</url></entry><entry path="stock.sql" revision="177"><commit revision="170"/></entry></info>'
        status = '<status><target path="."><entry path="stock.sql"><wc-status item="modified" props="normal" revision="177"/></entry></target></status>'
        return subprocess.CompletedProcess(command, 0, info if "info" in command else status, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    files, summary = read_svn(tmp_path)
    assert [command[1] for command in calls] == ["info", "status"]
    assert summary["availability"] == "available"
    assert summary["network_access"] is False
    assert files["stock.sql"]["revision"] == "177"
    assert files["stock.sql"]["local_status"] == "modified"


def test_svn_query_failure_is_explicit(tmp_path, monkeypatch):
    (tmp_path / ".svn").mkdir()

    def failed(*args, **kwargs):
        raise FileNotFoundError("svn unavailable")

    monkeypatch.setattr(subprocess, "run", failed)
    assert read_svn(tmp_path)[1]["reason"] == "local-svn-query-failed"


def test_empty_svn_info_is_unavailable(tmp_path):
    files, summary = parse_svn("<info/>", "<status/>", tmp_path)
    assert files == {}
    assert summary["availability"] == "unavailable"
