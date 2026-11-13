import os

import pytest

from apollo_rag.inputs import collect_inputs, read_document, secret_bearing


def put(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_readable_forms_analysis_sql_retained_but_generated_and_vendor_data_rejected(tmp_path):
    good = "forms_10g/analysis/APOLLO/sql/triggers/stock_trigger.sql"
    put(tmp_path, good, "-- original trigger\nbegin post_stock; end;")
    for name in ("forms_10g/analysis/inventory.json", "forms_10g/analysis/generated.json",
                 "webix/webix.js", "app.min.js", "config.json", ".env", "trace.har", ".svn/wc.db"):
        put(tmp_path, name, "not part of the corpus")
    documents, excluded = collect_inputs([], tmp_path)
    assert [item.path for item in documents] == [good]
    reasons = {item["path"]: item["reason"] for item in excluded}
    assert reasons["forms_10g/analysis/generated.json"] == "generated-analysis-json"
    assert reasons["webix"] == "excluded-directory"
    assert reasons["config.json"] == "configuration-or-credentials"
    assert reasons[".env"] == "configuration-or-credentials"


def test_vendor_and_agent_directories_are_excluded_but_custom_application_is_kept(tmp_path):
    custom = "application/apForm.js"
    put(tmp_path, custom, "function postStock() { return 1; }")
    directories = ("webix-7.3.0", "WEBIX_7_2_7", "codebase_7_2_7", ".cursor", "awesome-codex-subagents")
    for directory in directories:
        put(tmp_path, f"{directory}/readable.md", "bundled vendor or agent documentation")
    documents, excluded = collect_inputs([], tmp_path)
    assert [document.path for document in documents] == [custom]
    assert {item["path"]: item["reason"] for item in excluded} == {
        directory: "excluded-directory" for directory in directories
    }


@pytest.mark.parametrize("text", [
    "password = 'private-fixture-literal-974'", '{"api_key": "private-fixture-literal-974"}',
    "oracle://admin:private-fixture-literal-974@localhost/db", "-----BEGIN PRIVATE KEY-----",
    "connect admin/private-fixture-literal-974@localdb", "alter user admin identified by unsafe_literal;",
    'CREATE USER "fixture user"\nIDENTIFIED BY "private-fixture-literal-974";',
    "CREATE DATABASE LINK fixture_link CONNECT TO fixture_user IDENTIFIED BY unsafe_literal;",
    "ALTER DATABASE LINK fixture_link CONNECT TO fixture_user IDENTIFIED BY unsafe_literal;",
    "CREATE PUBLIC DATABASE LINK fixture_link\nCONNECT TO fixture_user\nIDENTIFIED BY unsafe_literal;",
    "<connection><password>private-fixture-literal-974</password></connection>",
    '<connection><auth:apiKey xmlns:auth="urn:fixture">private-fixture-literal-974</auth:apiKey></connection>',
    "<password><![CDATA[fixture-private-value-974]]></password>",
    "<password>\n<![CDATA[\nfixture-private-value-974\n]]>\n</password>",
    '<auth:apiKey xmlns:auth="urn:fixture"><![CDATA[fixture-private-value-974]]></auth:apiKey>',
    "<password><value>fixture-private-value-974</value></password>",
    "<password>${DB_PASSWORD}fixture-private-value-974</password>",
    "<password>TODO: fixture-private-value-974</password>",
])
def test_credential_bearing_content_is_detected(text):
    assert secret_bearing(text)


@pytest.mark.parametrize("text", [
    "password = ''", "password = os.getenv('DB_PASSWORD')", "api_key: <your-key>",
    "select stock_account from ledger;", "password = :password",
    "<connection><password></password></connection>",
    "<connection><password>TODO: load from env or secrets vault</password></connection>",
    "<password> \n </password>",
    "<password>${DB_PASSWORD}</password>",
    "<password> $DB_PASSWORD </password>",
    "Locations are identified by their code.",
    "One financial item is identified by its account.",
    "These software components are identified by\nunique codes.",
    "Create user documentation; locations are identified by their code.",
])
def test_nonliteral_placeholders_and_business_sql_remain_eligible(text):
    assert not secret_bearing(text)


def test_size_binary_encoding_and_cp1250_provenance(tmp_path):
    binary = tmp_path / "binary.sql"
    binary.write_bytes(b"select\x00secret")
    assert read_document(binary, "source", binary.name, 100)[1]["reason"] == "binary-content"
    assert read_document(binary, "source", binary.name, 2)[1]["reason"] == "file-size-limit"
    legacy = tmp_path / "legacy.sql"
    legacy.write_bytes("-- čćžšđ\nselect 1;".encode("cp1250"))
    document, error = read_document(legacy, "source", legacy.name, 100)
    assert error is None
    assert document.encoding == "cp1250"
    assert "čćžšđ" in document.text
    assert read_document(legacy, "manual", legacy.name, 100)[1]["reason"] == "unsupported-text-encoding"


def test_total_limit_does_not_silently_truncate_documents(tmp_path):
    put(tmp_path, "one.sql", "select 1;" * 10)
    put(tmp_path, "two.sql", "select 2;" * 10)
    with pytest.raises(ValueError, match="max-total-bytes"):
        collect_inputs([], tmp_path, max_total_bytes=100)


def test_symlink_cannot_escape_source_root(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    outside = put(tmp_path, "outside.sql", "outside private content")
    try:
        os.symlink(outside, root / "escape.sql")
    except OSError:
        pytest.skip("Creating symlinks requires OS permission")
    documents, excluded = collect_inputs([], root)
    assert not documents
    assert excluded[0]["reason"] == "symlink-or-junction"
