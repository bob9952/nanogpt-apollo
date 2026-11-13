"""Load only tokenizer files; never load model weights or remote Python code."""

from importlib.metadata import version
import os
from pathlib import Path

MODEL_ID = "intfloat/multilingual-e5-small"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"


class E5Tokenizer:
    def __init__(self, tokenizer, identity):
        self.tokenizer = tokenizer
        self.identity = identity

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))


def load_tokenizer(cache_dir=DEFAULT_CACHE, revision="main", offline=False):
    """Resolve the requested revision to a cache snapshot and record its commit."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("USE_TORCH", "0")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        cache_dir = Path(cache_dir).resolve()
        config = Path(hf_hub_download(
            MODEL_ID, "tokenizer_config.json", revision=revision,
            cache_dir=str(cache_dir), local_files_only=offline,
        ))
        resolved_revision = config.parent.name
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=resolved_revision, cache_dir=str(cache_dir),
            local_files_only=offline, trust_remote_code=False, use_fast=True,
        )
        if not tokenizer.is_fast:
            raise ValueError("The E5 fast tokenizer is required")
    except Exception as exc:
        # Do not echo hub/HTTP exceptions: they can contain proxy credentials.
        raise RuntimeError(
            "E5 tokenizer unavailable. Cache intfloat/multilingual-e5-small with "
            "an approved online prepare run, or provide --tokenizer-cache pointing "
            "to its Hugging Face cache. Use --offline only after caching it. "
            f"Failure type: {type(exc).__name__}."
        ) from None
    return E5Tokenizer(tokenizer, {
        "model": MODEL_ID, "requested_revision": revision,
        "revision": resolved_revision, "transformers": version("transformers"),
        "tokenizers": version("tokenizers"), "special_tokens_included": True,
    })
