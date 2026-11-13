"""Explicit-vector HTTP client; Chroma never chooses or downloads a model."""

from contextlib import contextmanager
import re

import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError
import httpx


@contextmanager
def chroma_errors():
    try:
        yield
    except (ChromaError, httpx.HTTPError):
        raise RuntimeError("Chroma request failed; check docker compose logs chroma, server readiness, and --chroma-host/--chroma-port") from None
    except (ValueError, RuntimeError):
        raise
    except Exception:
        # The SDK uses plain Exception for unrecognized HTTP error responses.
        raise RuntimeError("Chroma returned an unexpected response; inspect docker compose logs chroma") from None


def connect(host, port):
    if (not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", host)
            or not isinstance(port, int) or not 1 <= port <= 65535):
        raise ValueError("Use a Chroma hostname without a URL/path and a port between 1 and 65535")
    try:
        return chromadb.HttpClient(host=host, port=port,
                                   settings=Settings(anonymized_telemetry=False,
                                                     chroma_otel_granularity="none"))
    except Exception:
        # HttpClient wraps some connection failures in ValueError with server text.
        raise RuntimeError("Chroma connection failed; start docker compose up -d chroma and check --chroma-host/--chroma-port") from None
