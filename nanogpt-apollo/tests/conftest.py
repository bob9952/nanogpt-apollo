import pytest


class FixtureTokenizer:
    """Deterministic, deliberately strict counter; it never represents E5."""

    identity = {"model": "fixture-character-counter", "revision": "fixture-v1",
                "special_tokens_included": True}

    def count(self, text):
        return len(text) + 2


@pytest.fixture
def tokenizer():
    return FixtureTokenizer()
