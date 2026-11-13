import os

import pytest

from apollo_rag.chunks import document_chunks
from apollo_rag.inputs import SourceDocument, sha256
from apollo_rag.tokenizer import DEFAULT_CACHE, MODEL_ID, load_tokenizer


@pytest.mark.integration
def test_real_cached_e5_tokenizer_enforces_384_including_context_and_special_tokens():
    if os.environ.get("APOLLO_REAL_TOKENIZER") != "1":
        pytest.skip("Set APOLLO_REAL_TOKENIZER=1 after explicitly caching the public E5 tokenizer")
    tokenizer = load_tokenizer(os.environ.get("APOLLO_TOKENIZER_CACHE", DEFAULT_CACHE), offline=True)
    text = "# Stock accounts / Konta zaliha\n\n" + "Odaberite konto zaliha prije knjiženja. 仓库库存账目. " * 180
    document = SourceDocument("manual", "fixture.txt", "/local/fixture.txt", sha256(text.encode()), "utf-8", text)
    chunks, _ = document_chunks(document, tokenizer, {})
    assert len(chunks) > 1
    assert tokenizer.identity["model"] == MODEL_ID
    assert len(tokenizer.identity["revision"]) == 40
    for chunk in chunks:
        assert len(tokenizer.tokenizer.encode(chunk["embedding_text"], add_special_tokens=True)) == chunk["token_count"]
        assert chunk["token_count"] <= 384
    assert chunks[-1]["text"].endswith("仓库库存账目.")
