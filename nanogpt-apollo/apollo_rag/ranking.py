"""Lexical scoring and reciprocal rank fusion for existing Chroma collections."""

from collections import Counter
from math import log1p


RANKERS = ("legacy", "dense", "bm25", "hybrid", "routed")


def reciprocal_rank_fusion(*rankings):
    scores = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0.0) + 1 / (60 + rank)
    return scores


def bm25(documents, query_terms):
    """Score a collection of token lists using document frequency and length."""
    if not documents or not query_terms:
        return {}
    average_length = sum(map(len, documents.values())) / len(documents)
    if average_length == 0:
        return {}

    frequencies = {identifier: Counter(terms) for identifier, terms in documents.items()}
    idf = {}
    for term in sorted(query_terms):
        count = sum(term in counts for counts in frequencies.values())
        idf[term] = log1p((len(documents) - count + 0.5) / (count + 0.5))

    scores = {}
    for identifier, counts in frequencies.items():
        length_weight = 1.2 * (0.25 + 0.75 * len(documents[identifier]) / average_length)
        score = sum(weight * 2.2 * counts[term] / (counts[term] + length_weight)
                    for term, weight in idf.items() if counts[term])
        if score:
            scores[identifier] = score
    return scores
