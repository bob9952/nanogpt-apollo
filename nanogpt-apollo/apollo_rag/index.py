"""Store explicit Ollama E5 embeddings in a resumable Chroma collection."""

import numpy as np

from chromadb.errors import NotFoundError

from .bundle import load_bundle
from .chroma import chroma_errors, connect
from .ollama import ContextTooLong, DIMENSION, EMBED_MODEL, normalized_vector

SCHEMA = 2
ALGORITHM = "e5-prefix-exact-or-character-weighted-split-v1"
KINDS = {"manual", "source", "both"}


def check_kind(kind):
    if kind not in KINDS:
        raise ValueError("--kind must be manual, source, or both")


def collection_settings(manifest, digest, kind, count):
    return {"schema_version": SCHEMA, "algorithm": ALGORITHM,
            "corpus_fingerprint": manifest["corpus_fingerprint"],
            "embedding_model": EMBED_MODEL, "embedding_digest": digest,
            "dimension": DIMENSION, "kind": kind, "chunk_count": count}


def check_collection(collection, expected):
    info = collection.metadata or {}
    if any(info.get(key) != value for key, value in expected.items()):
        raise ValueError("Chroma collection belongs to a different corpus, scope, or E5 model; index with a new --collection name")
    return info


class PassageEmbeddings:
    def __init__(self, ollama):
        self.ollama = ollama
        self.requests = 0

    def embed_batch(self, texts):
        try:
            self.requests += 1
            return self.ollama.embed(texts)
        except ContextTooLong:
            if len(texts) > 1:
                middle = len(texts) // 2
                return self.embed_batch(texts[:middle]) + self.embed_batch(texts[middle:])
            return [self.split_passage(texts[0])]

    def split_passage(self, text):
        if not text.startswith("passage: "):
            raise ValueError("Only prepared passage inputs can use split recovery")
        # Keep every character and the original passage/citation. Only embedding
        # inputs are split; child vectors are pooled by their character lengths.
        budget = [128]

        def leaves(body, depth):
            if depth > 12 or len(body) < 32 or budget[0] <= 0:
                raise RuntimeError("E5 passage still exceeds context after bounded splitting; prepare smaller passages")
            middle = len(body) // 2
            result = []
            for part in (body[:middle], body[middle:]):
                if budget[0] <= 0:
                    raise RuntimeError("E5 split attempt limit reached; prepare smaller passages")
                budget[0] -= 1
                try:
                    self.requests += 1
                    vector = self.ollama.embed(["passage: " + part])[0]
                except ContextTooLong:
                    result.extend(leaves(part, depth + 1))
                else:
                    result.append((vector, len(part)))
            return result

        parts = leaves(text[len("passage: "):], 0)
        return normalized_vector(sum(vector.astype(np.float64) * weight for vector, weight in parts))


def build_index(corpus, collection, ollama, *, batch_size=16, kind="both",
                progress=None, chroma_host="127.0.0.1", chroma_port=8000):
    check_kind(kind)
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
        raise ValueError("--batch-size must be between 1 and 128")
    manifest, all_chunks = load_bundle(corpus)
    chunks = [chunk for chunk in all_chunks if kind == "both" or chunk["kind"] == kind]
    if not chunks:
        raise ValueError("Corpus has no chunks for --kind; choose another scope")
    digest = ollama.embedding_digest()
    settings = collection_settings(manifest, digest, kind, len(chunks))
    with chroma_errors():
        client = connect(chroma_host, chroma_port)
        try:
            target = client.get_collection(collection, embedding_function=None)
        except NotFoundError:
            target = client.create_collection(
                collection, embedding_function=None, metadata={**settings, "complete": False},
                configuration={"hnsw": {"space": "cosine"}})
        check_collection(target, settings)
        target.modify(metadata={**settings, "complete": False})
        embeddings = PassageEmbeddings(ollama)
        reused = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            existing = target.get(ids=[chunk["id"] for chunk in batch], include=[])
            known = set(existing["ids"])
            pending = [chunk for chunk in batch if chunk["id"] not in known]
            reused += len(known)
            if pending:
                vectors = embeddings.embed_batch([chunk["embedding_text"] for chunk in pending])
                # Do not publish a batch made while the model was being replaced.
                if ollama.embedding_digest() != digest:
                    raise ValueError("E5 digest changed during indexing; use a stable model and a new --collection name")
                target.upsert(
                    ids=[chunk["id"] for chunk in pending],
                    embeddings=[vector.tolist() for vector in vectors],
                    documents=[chunk["text"] for chunk in pending],
                    metadatas=[{"kind": chunk["kind"], "path": chunk["path"],
                                "sha256": chunk["sha256"]} for chunk in pending])
            if progress:
                progress(min(start + batch_size, len(chunks)), len(chunks))
        if ollama.embedding_digest() != digest:
            raise ValueError("E5 digest changed during indexing; use a stable model and a new --collection name")
        if target.count() != len(chunks):
            raise ValueError("Chroma collection contains unexpected records; index with a new --collection name")
        target.modify(metadata={**settings, "complete": True})
        return {**settings, "complete": True, "embedding_requests": embeddings.requests, "reused": reused}
