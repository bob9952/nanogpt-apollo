"""Query Chroma cosine rankings and cite the original validated passages."""

import re
from time import perf_counter

from chromadb.errors import NotFoundError

from .bundle import load_bundle
from .chroma import chroma_errors, connect
from .index import check_collection, check_kind, collection_settings
from .ollama import ContextTooLong


def manual_heading_only(chunk):
    section = chunk.get("section")
    return (chunk["kind"] == "manual" and bool(section)
            and " ".join(chunk["text"].split()) == " ".join(section.split()))


def reciprocal_ranks(vector_ids, literal_ids, *, constant=60):
    scores = {}
    for ranking in (vector_ids, literal_ids):
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (constant + rank)
    return scores


def search(corpus, collection, query, ollama, *, top_k=3, kind="both",
           chroma_host="127.0.0.1", chroma_port=8000):
    started = perf_counter()
    check_kind(kind)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Supply a nonempty search query")
    if len(query) > 8192:
        raise ValueError("Search query is too long; shorten it to one focused question")
    if not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("--top-k must be between 1 and 100")
    manifest, chunks = load_bundle(corpus)
    digest = ollama.embedding_digest()
    with chroma_errors():
        client = connect(chroma_host, chroma_port)
        try:
            target = client.get_collection(collection, embedding_function=None)
        except NotFoundError:
            raise ValueError("Chroma collection is missing; run index with this --collection name first") from None
        info = target.metadata or {}
        scope = info.get("kind")
        if scope not in {"manual", "source", "both"}:
            raise ValueError("Chroma collection scope is incompatible; index with a new --collection name")
        expected = {chunk["id"]: chunk for chunk in chunks if scope == "both" or chunk["kind"] == scope}
        check_collection(target, collection_settings(manifest, digest, scope, len(expected)))
        if info.get("complete") is not True or target.count() != len(expected):
            raise ValueError("Chroma collection is incomplete; rerun index with the same corpus, kind, and --collection to resume")
        if scope != "both" and kind != scope:
            raise ValueError("Search --kind exceeds this collection scope; use its kind or index a new collection with --kind both")
        selected_count = sum(kind == "both" or chunk["kind"] == kind for chunk in expected.values())
        results = []
        if selected_count:
            try:
                vector = ollama.embed(["query: " + query])[0]
            except ContextTooLong:
                raise ValueError("Query exceeds the E5 context limit; shorten the query; truncation is disabled") from None
            if ollama.embedding_digest() != digest:
                raise ValueError("E5 digest changed during search; retry with a stable model")
            candidate_count = min(max(100, top_k * 20), selected_count)
            while True:
                ranking = target.query(query_embeddings=[vector.tolist()],
                                       where={"kind": kind} if kind != "both" else None,
                                       n_results=candidate_count, include=["distances"])
                vector_scores = {}
                for identifier, distance in zip(ranking["ids"][0], ranking["distances"][0], strict=True):
                    if identifier not in expected or (kind != "both" and expected[identifier]["kind"] != kind):
                        raise ValueError("Chroma returned an unexpected passage; index with a new --collection name")
                    if manual_heading_only(expected[identifier]):
                        continue
                    vector_scores[identifier] = 1.0 - distance
                if len(vector_scores) >= top_k or candidate_count == selected_count:
                    break
                # Headings can fill the first candidate window in existing bundles.
                candidate_count = min(candidate_count * 2, selected_count)
            # Literal Apollo identifiers supplement semantic retrieval without a
            # second index. Ordinary words and bare numbers stay semantic only.
            identifiers = {word.casefold() for word in re.findall(r"\w+", query)
                           if any(char.isalpha() for char in word)
                           and ("_" in word or any(char.isdigit() for char in word))}
            literal_hits = []
            if identifiers:
                for identifier, chunk in expected.items():
                    if (kind != "both" and chunk["kind"] != kind) or manual_heading_only(chunk):
                        continue
                    words = set(re.findall(r"\w+", (chunk["path"] + " " + chunk["text"]).casefold()))
                    matches = len(identifiers.intersection(words))
                    if matches:
                        literal_hits.append((identifier, matches))
            literal_ids = [identifier for identifier, _ in sorted(literal_hits, key=lambda hit: (-hit[1], hit[0]))][:100]
            scores = reciprocal_ranks(list(vector_scores), literal_ids)
            ranked = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:top_k]
            results = [{**expected[identifier], "score": scores[identifier],
                        "vector_score": vector_scores.get(identifier)} for identifier in ranked]
        return {"query": query, "kind": kind, "results": results,
                "elapsed_seconds": perf_counter() - started,
                "corpus_fingerprint": manifest["corpus_fingerprint"], "embedding_digest": digest}
