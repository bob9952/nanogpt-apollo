"""Small Ollama client: explicit models and requests, never downloads."""

import json
from urllib.parse import urlsplit

import httpx
import numpy as np

DEFAULT_URL = "http://localhost:11434"
CHAT_MODEL = "qwen3.5:4b-q4_K_M"
EMBED_MODEL = "qllama/multilingual-e5-small:q8_0"
DIMENSION = 384
NUM_CTX = 4096
NUM_PREDICT = 256


class ContextTooLong(RuntimeError):
    pass


def normalized_vector(value):
    try:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (DIMENSION,) or not np.isfinite(vector).all():
            raise ValueError
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise RuntimeError("Invalid E5 embedding; expected 384 finite, nonzero dimensions") from None
    return vector / norm


class Ollama:
    def __init__(self, endpoint=DEFAULT_URL, *, chat_model=CHAT_MODEL, num_predict=NUM_PREDICT, transport=None):
        try:
            url = urlsplit(endpoint)
            valid = (url.scheme in {"http", "https"} and url.hostname and url.port != 0
                     and not url.username and not url.password and not url.query
                     and not url.fragment and url.path in {"", "/"})
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Use an http(s) Ollama root URL without credentials, query, or path")
        if type(num_predict) is not int or not 1 <= num_predict < NUM_CTX:
            raise ValueError(f"--num-predict must be between 1 and {NUM_CTX - 1} tokens")
        # Explicit endpoint only: never send proprietary text through an env proxy.
        self.chat_model = chat_model
        self.num_predict = num_predict
        self.client = httpx.Client(base_url=endpoint.rstrip("/"), timeout=httpx.Timeout(120, connect=3),
                                   trust_env=False, transport=transport)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def request(self, method, path, payload=None):
        try:
            response = self.client.request(method, path, json=payload)
        except httpx.HTTPError:
            raise RuntimeError("Ollama is unreachable; start Ollama and check --ollama-url") from None
        if response.is_error:
            # Inspect only to classify; never expose response bodies or HTTP errors.
            body = response.text.lower()
            if path == "/api/embed" and response.status_code in {400, 413, 422, 500} and (
                "context length" in body or "input length" in body or "too long" in body
                or response.status_code == 413
            ):
                raise ContextTooLong("Ollama rejected an embedding input length")
            raise RuntimeError("Ollama request failed; check installed models and Ollama server logs")
        try:
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError
            return result
        except (ValueError, TypeError):
            raise RuntimeError("Ollama returned invalid JSON; check the server version") from None

    def models(self):
        records = self.request("GET", "/api/tags").get("models")
        if not isinstance(records, list):
            raise RuntimeError("Ollama model inventory is invalid; check the server version")
        result = {}
        for record in records:
            if not isinstance(record, dict):
                raise RuntimeError("Ollama model inventory is invalid; check the server version")
            if record.get("remote_host") or record.get("remote_model"):
                continue
            name, digest = record.get("name", record.get("model")), record.get("digest")
            if isinstance(name, str) and isinstance(digest, str) and digest:
                result[name] = digest
        return result

    def embedding_digest(self):
        digest = self.models().get(EMBED_MODEL)
        if not digest:
            raise RuntimeError(f"Required embedding model is missing; install {EMBED_MODEL} explicitly")
        return digest

    def doctor(self):
        version = self.request("GET", "/api/version").get("version")
        if not isinstance(version, str) or not version:
            raise RuntimeError("Ollama version response is invalid; check the server version")
        models = self.models()
        required = (self.chat_model, EMBED_MODEL)
        missing = [name for name in required if name not in models]
        if missing:
            raise RuntimeError("Required Ollama model missing; install explicitly: " + ", ".join(missing))
        return {"version": version, "models": {name: models[name] for name in required}}

    def embed(self, texts):
        result = self.request("POST", "/api/embed", {
            "model": EMBED_MODEL, "input": texts, "truncate": False,
        }).get("embeddings")
        if not isinstance(result, list) or len(result) != len(texts):
            raise RuntimeError("Ollama returned the wrong embedding count; check the server version")
        return [normalized_vector(value) for value in result]

    def generate(self, prompt, system):
        """Yield validated NDJSON records; never expose server bodies in errors."""
        payload = {
            "model": self.chat_model, "prompt": prompt, "system": system,
            "think": False, "stream": True, "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": NUM_CTX,
                        "num_thread": 8, "num_predict": self.num_predict},
        }
        done = False
        line_number = 0
        try:
            with self.client.stream("POST", "/api/generate", json=payload,
                                    timeout=httpx.Timeout(300, connect=3)) as response:
                if response.is_error:
                    raise RuntimeError(f"Ollama generation HTTP {response.status_code}; check installed chat model and server logs")
                for line_number, line in enumerate(response.iter_lines(), 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except (ValueError, TypeError):
                        raise RuntimeError(f"Ollama generation malformed JSON at stream line {line_number}; inspect server logs") from None
                    if not isinstance(record, dict):
                        raise RuntimeError(f"Ollama generation invalid record at stream line {line_number}; expected an object")
                    if "error" in record:
                        raise RuntimeError(f"Ollama generation server error at stream line {line_number}; inspect server logs")
                    if (not isinstance(record.get("done"), bool)
                            or not isinstance(record.get("response", ""), str)):
                        raise RuntimeError(f"Ollama generation invalid fields at stream line {line_number}; inspect server version")
                    if done:
                        raise RuntimeError(f"Ollama generation record after completion at stream line {line_number}; inspect server version")
                    done = record["done"]
                    # Deliberately do not forward thinking or arbitrary server fields.
                    yield {key: record[key] for key in (
                        "response", "done", "done_reason", "load_duration", "eval_count", "eval_duration"
                    ) if key in record}
        except httpx.HTTPError as error:
            raise RuntimeError(f"Ollama generation transport failure ({type(error).__name__}) after stream line {line_number}; check endpoint and server logs") from None
        if not done:
            raise RuntimeError(f"Ollama generation incomplete after stream line {line_number}; missing done record; retry or inspect server logs")
