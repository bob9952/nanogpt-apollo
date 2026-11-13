import json

import httpx
import pytest

from apollo_rag.ollama import CHAT_MODEL, DIMENSION, EMBED_MODEL, ContextTooLong, Ollama


def vector(coordinate=0):
    return [1.0 if i == coordinate else 0.0 for i in range(DIMENSION)]


def test_doctor_checks_version_exact_tags_and_digests_without_pulling():
    seen = []

    def respond(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"version": "0.12.fixture"} if request.url.path == "/api/version" else {
            "models": [{"name": name, "digest": "fixture-" + str(i)}
                       for i, name in enumerate((CHAT_MODEL, EMBED_MODEL))]})

    with Ollama(transport=httpx.MockTransport(respond)) as client:
        status = client.doctor()
    assert status["models"][EMBED_MODEL] == "fixture-1"
    assert seen == [("GET", "/api/version"), ("GET", "/api/tags")]


def test_missing_exact_tag_is_actionable():
    def respond(request):
        return httpx.Response(200, json={"version": "fixture", "models": [
            {"name": "qllama/multilingual-e5-small:latest", "digest": "wrong-tag"}]})

    with Ollama(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(RuntimeError, match="install explicitly"):
            client.doctor()
        with pytest.raises(RuntimeError, match="Required embedding model"):
            client.embedding_digest()


def test_embed_sends_exact_input_array_without_truncation():
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [vector(), vector(1)]})

    with Ollama(transport=httpx.MockTransport(respond)) as client:
        values = client.embed(["passage: Original\n input", "query: stock account"])
    assert seen == [{"model": EMBED_MODEL, "input": ["passage: Original\n input", "query: stock account"], "truncate": False}]
    assert values[1][1] == 1


@pytest.mark.parametrize("payload", [
    {"embeddings": []}, {"embeddings": [[1, 2]]}, {"embeddings": [[0] * DIMENSION]},
    {"embeddings": "invalid"}, {"embeddings": [["invalid"] * DIMENSION]},
])
def test_invalid_embeddings_are_rejected(payload):
    with Ollama(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
        with pytest.raises(RuntimeError, match="embedding"):
            client.embed(["query: test"])


def test_context_error_is_classified_without_echoing_response():
    with Ollama(transport=httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": "input length exceeds context length private-fixture-text"}))) as client:
        with pytest.raises(ContextTooLong) as error:
            client.embed(["passage: test"])
    assert "private-fixture-text" not in str(error.value)


@pytest.mark.parametrize("endpoint", ["http://user:private@localhost:11434", "file:///private", "http://localhost/api", "http://localhost?private=1"])
def test_endpoint_rejects_credentials_and_nonroot_urls(endpoint):
    with pytest.raises(ValueError, match="without credentials"):
        Ollama(endpoint)


def test_transport_and_server_errors_do_not_echo_sensitive_details():
    def unavailable(request):
        raise httpx.ConnectError("http://proxy:private@host", request=request)

    with Ollama(transport=httpx.MockTransport(unavailable)) as client:
        with pytest.raises(RuntimeError, match="unreachable") as error:
            client.models()
    assert "private" not in str(error.value)
    with Ollama(transport=httpx.MockTransport(lambda request: httpx.Response(500, text="private-server-body"))) as client:
        with pytest.raises(RuntimeError) as error:
            client.models()
    assert "private-server-body" not in str(error.value)


@pytest.mark.parametrize("payload", [[], None, "private-body"])
def test_nonobject_json_is_rejected(payload):
    with Ollama(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
        with pytest.raises(RuntimeError, match="invalid JSON"):
            client.models()
