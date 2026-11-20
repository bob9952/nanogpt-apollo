"""Query Chroma cosine rankings and cite the original validated passages."""

from pathlib import PurePosixPath
import re
from time import perf_counter

from chromadb.errors import NotFoundError

from .bundle import load_bundle
from .chroma import chroma_errors, connect
from .index import check_collection, check_kind, collection_settings
from .ollama import ContextTooLong
from .ranking import RANKERS, bm25, reciprocal_rank_fusion
from .support import BALANCE_FOCUS, STOCK_FOCUS, reviewed_passages


def manual_heading_only(chunk):
    section = chunk.get("section")
    return (chunk["kind"] == "manual" and bool(section)
            and " ".join(chunk["text"].split()) == " ".join(section.split()))


# These aliases connect frequent inventory terms in the Serbian and English manuals.
WORD_ALIASES = {
    word: term
    for term, variants in (
        ("account", "accounts ledger ledgers konto konta kontu kontom kontima"),
        ("stock", "inventory zalih zaliha zalihe zalihu zalihama"),
        ("warehouse", "warehouses magacin magacina magacinu magacinom magacini magacine magacinima skladiste skladista skladistu"),
        ("item", "items artikal artikla artiklu artiklom artikli artikle artikala artiklima"),
        ("default", "defaults podrazumevan podrazumevana podrazumevane podrazumevani podrazumevano"),
    )
    for word in variants.split()
}
IGNORED_WORDS = frozenset(
    "and are can for how the this what where which why with da gde gdje ili "
    "kako koji koje mogu moze nije ovo se sta sto za zasto proverava podesava".split()
)
SERBIAN_LETTERS = str.maketrans({"\u010d": "c", "\u0107": "c", "\u0161": "s", "\u017e": "z", "\u0111": "dj"})


CYRILLIC_TO_LATIN = str.maketrans(dict(zip(
    "абвгдђежзијклљмнњопрстћуфхцчџш",
    "a b v g d dj e z z i j k l lj m n nj o p r s t c u f h c c dz s".split(),
    strict=True,
)))


def normalized_words(text):
    folded = text.casefold().translate(CYRILLIC_TO_LATIN).translate(SERBIAN_LETTERS)
    words = re.findall(r"\w+", folded)
    return [word for word in words if len(word) >= 2 and any(char.isalpha() for char in word)]


def search_words(text):
    return set(normalized_words(text))


def candidate_scores(chunks, vector_ids, query, kind):
    words = search_words(query) - IGNORED_WORDS
    codes = {word for word in words if "_" in word or any(char.isdigit() for char in word)}
    keywords = {WORD_ALIASES.get(word, word) for word in words - codes}
    # Keep semantic rank as a tie-breaker. A form-code match outranks any
    # keyword-only match, and a keyword match outranks a vector-only result.
    scores = {identifier: 1 / (60 + rank) for rank, identifier in enumerate(vector_ids, 1)}
    if not words:
        return scores

    for identifier, chunk in chunks.items():
        if (kind != "both" and chunk["kind"] != kind) or manual_heading_only(chunk):
            continue
        body_words = search_words(chunk["text"])
        matched_codes = codes.intersection(body_words | search_words(chunk["path"])) if codes else set()
        matched_terms = keywords.intersection(WORD_ALIASES.get(word, word) for word in body_words)
        strength = len(matched_codes) * (len(keywords) + 1) + len(matched_terms)
        if strength:
            scores[identifier] = strength + scores.get(identifier, 0)
    return scores


FORM_CODE = re.compile(r"[a-z]{2,}[0-9]{2,}[a-z][0-9]{2,}")
BALANCE_SYMPTOM = re.compile(
    r"\b(?:neravnotez\w*|(?:nije|nisu) (?:mi )?(?:ravnotezi|izbalansiran\w*|uravnotezen\w*))\b"
)


def support_focus(query):
    words = normalized_words(query)
    terms = {WORD_ALIASES.get(word, word) for word in words}
    if {"account", "stock"} <= terms and BALANCE_SYMPTOM.search(" ".join(words)):
        return (
            (STOCK_FOCUS, "Konto zaliha artikla, podrazumevani konto po magacinu i tipu artikla LIN001M001"),
            (BALANCE_FOCUS, "Neravnoteža dokumenta pri knjiženju, duguje i potražuje. "
             "Financial balance Incorrect debit side is not equal credit side."),
        )
    return ()


def lexical_terms(text):
    terms = [WORD_ALIASES.get(word, word) for word in normalized_words(text)
             if word not in IGNORED_WORDS]
    # Serbian "konto zaliha" and English "stock account" reverse word order.
    pairs = [" ".join(sorted((left, right))) for left, right in zip(terms, terms[1:])]
    return terms + pairs


def exact_matches(chunks, query):
    keys = {word for word in search_words(query) if "_" in word or FORM_CODE.fullmatch(word)}
    if not keys:
        return {}
    return {identifier: len(keys & search_words(f"{chunk['path']} {chunk['section'] or ''} {chunk['text']}"))
            for identifier, chunk in chunks.items()}


# Include only document scopes known from the corpus inventory. An unknown
# document or a source checkout does not establish a live configuration.
MANUAL_SCOPES = {
    "apollo g4 erp users guide v2.txt": "opšte uputstvo za G4",
    "specification of app english.txt": "opšta specifikacija",
    "lovcen auto v1.0.pdf.txt": "uputstvo za konkretnu instalaciju",
    "apollog4 uputstvo - lovcen auto offline jul 2010.txt": "uputstvo za konkretnu instalaciju",
    "uputstvopostacg.txt": "uputstvo za konkretnu instalaciju",
    "g5 igorl.txt": "uputstvo za konkretnu instalaciju",
    "privredni-savetnik-2015-02-ocr.txt": "stručni članak iz 2015.",
    "privredni-savetnik-2022-05-e-fakturisanje.txt": "stručni članak iz 2022.",
}
ROUTING_STOPWORDS = IGNORED_WORDS | frozenset(
    "mi moj moja mom mojoj sam smo ne nema nisu nisam je su dajte reci sada prvo "
    "trazim gresku problem proverim provjerim radim naci nadjem nadem nasao nasla "
    "zelim uputis vidim proveriti provjeriti uporedim opcije koliko stane".split()
)
ROUTING_ALIASES = {
    word: term
    for term, variants in (
        ("balance", "ravnotezi ravnoteza ravnoteze neravnoteza neravnotezi neravnoteze izbalansiran uravnotezen"),
        ("debit", "duguje dugovna dugovno dugovnu dugovni dugovnim"),
        ("credit", "potrazuje potrazna potrazno potraznu potrazni potraznim"),
        ("post", "posting knjizi knjizim knjizenje knjizenju knjizenja proknjizena proknjizen"),
        ("document", "dokument dokumenta dokumentu dokumentom documents dokumenti dokumentima"),
        ("system", "sistem sistema sistemu"),
        ("financial", "finansijski finansijska finansijsku finansijske finansijskog"),
        ("blank", "prazno prazan prazna izostavi izostavljen blank empty omitted"),
        ("state", "stanje stanja stanju saldo balances"),
        ("type", "tip tipa tipu types"),
        ("receipt", "prijemnica prijemnice prijemnicu prijemnici receipts"),
        ("dispatch", "otpremnica otpremnice otpremnicu otpremnici"),
        ("expiration", "vaznost vaznosti isticanje expiry"),
        ("partner", "partneru partnera partnerom partners"),
        ("rule", "rules pravilo pravila pravilom"),
        ("override", "izuzetak poseban posebna posebnom specific"),
    )
    for word in variants.split()
}


def passage_scope(passage):
    if passage["kind"] == "source":
        return "izvorni kod iz korpusa; primena u instalaciji nije potvrđena"
    name = PurePosixPath(passage["path"]).name.casefold()
    return MANUAL_SCOPES.get(name, "obuhvat uputstva nije potvrđen")


def route_terms(text):
    words = [ROUTING_ALIASES.get(word, WORD_ALIASES.get(word, word))
             for word in normalized_words(text) if word not in ROUTING_STOPWORDS]
    pairs = [" ".join(sorted((left, right))) for left, right in zip(words, words[1:])]
    return words + pairs


def query_topics(query):
    # Keep the user's wording for embeddings; use the Serbian fold only to
    # recognize conjunctions and compare clauses. Match spans on original text
    # by splitting punctuation first, then splitting normalized conjunctions.
    topics = []
    for sentence in re.split(r"[!?;]+|\.(?!\d)(?=\s|$)|,\s+", query):
        # Transliteration can expand a character. Split original conjunctions
        # after matching individual words, which preserves original offsets.
        words = list(re.finditer(r"\w+", sentence))
        start = 0
        comparison = False
        for left, right in zip(words, words[1:]):
            current, following = (
                word[0].lower().translate(CYRILLIC_TO_LATIN).translate(SERBIAN_LETTERS)
                for word in (left, right)
            )
            if current == "izmedju":
                comparison = True
                start = right.start()
            if (comparison and current in {"i", "ili"}) or (
                current == "i" and following in {"sistem", "nije", "nisu", "ne"}
            ) or (
                current in {"i", "a"} and following in {"gde", "gdje", "kako", "sta", "koji", "koje", "zasto"}
            ):
                topics.append(sentence[start:left.start()].strip())
                start = right.start()
                comparison = False
        topics.append(sentence[start:].strip())
    specific = {}
    for topic in topics:
        terms = frozenset(term for term in route_terms(topic)
                          if " " not in term and term not in {"system", "document", "post"})
        if terms:
            specific.setdefault(terms, topic)
    if len(specific) < 3 and any({"stock", "account", "item"} <= terms for terms in specific):
        original = set(route_terms(query))
        if "default" not in original and ({"balance", "post", "blank"} & original):
            specific[frozenset({"default", "stock", "account"})] = (
                "Podrazumevani konto zaliha po magacinu i tipu artikla"
            )
    return tuple(specific.values())[:3] or (query,)


def form_codes(text):
    return {word for word in normalized_words(text) if FORM_CODE.fullmatch(word) or "_" in word}


def containing_forms(chunks):
    """Carry a numbered form heading into its subsections."""
    active_path, active_depth, active_codes = None, 0, set()
    forms = {}
    for identifier, chunk in chunks.items():
        if chunk["path"] != active_path:
            active_path, active_depth, active_codes = chunk["path"], 0, set()
        section = chunk.get("section") or ""
        heading = re.match(r"^(\d+(?:\.\d+)+)\.?\s+(.+)$", section)
        if heading and "..." not in heading[2]:
            depth = heading[1].count(".") + 1
            codes = form_codes(heading[2])
            if codes or depth <= active_depth:
                active_depth, active_codes = depth, codes
        forms[identifier] = active_codes
    return forms


def topic_terms(topic):
    terms = set(route_terms(topic))
    if {"document", "balance"} <= terms:
        terms.update({"debit", "credit", "financial", "balance financial", "credit debit"})
    if {"stock", "account"} <= terms and "default" not in terms:
        terms.update({"rule", "blank", "state", "partner", "expiration"})
    return terms


def matching_facts(terms, body):
    if {"document", "balance"} <= terms and {"debit", "credit"} <= body:
        return len({"debit", "credit", "balance"} & body)
    if {"stock", "account"} <= terms and {"stock", "account"} <= body:
        if "default" in terms:
            return len({"default", "warehouse", "type"} & body)
        return 2 * ("blank" in body) + len({"rule", "state"} & body)
    if {"account", "blank"} <= terms and {"account", "blank"} <= body:
        return 2
    return 0


def routed_candidates(chunks, forms, topic, dense, lexical, limit):
    """Rank the supplied topic, using exact codes as a constraint when present."""
    codes = form_codes(topic)
    terms = topic_terms(topic)
    dense_ids = sorted(dense, key=lambda identifier: (-dense[identifier], identifier))[:limit]
    lexical_ids = sorted(lexical, key=lambda identifier: (-lexical[identifier], identifier))[:limit]
    scores = reciprocal_rank_fusion(dense_ids, lexical_ids)
    exact = {}
    if codes:
        for identifier, chunk in chunks.items():
            header = form_codes(f"{chunk['path']} {chunk.get('section') or ''}")
            body = form_codes(chunk["text"])
            section_matches = codes & (header | forms[identifier])
            matches = section_matches | (codes & body)
            if matches:
                exact[identifier] = (len(matches), len(section_matches))
                scores.setdefault(identifier, 0.0)
        scores = {identifier: scores[identifier] for identifier in exact}

    def order(identifier):
        chunk = chunks[identifier]
        section = set(route_terms(chunk.get("section") or ""))
        body = set(route_terms(chunk["text"]))
        return (*(-count for count in exact.get(identifier, (0, 0))), -matching_facts(terms, body),
                -(scores[identifier] + (0.003 if terms & section else 0)), identifier)

    return sorted(scores, key=order), scores


def routed_results(chunks, queries, vector_lists, *, top_k):
    documents = {identifier: route_terms(chunk["embedding_text"].removeprefix("passage: "))
                 for identifier, chunk in chunks.items() if not manual_heading_only(chunk)}
    eligible = {identifier: chunks[identifier] for identifier in documents}
    forms = containing_forms(chunks) if any(form_codes(topic) for topic in queries) else {}
    shortlist = max(20, top_k)
    lists, scores, focuses = [], {}, {}
    for topic, vectors in zip(queries, vector_lists, strict=True):
        terms = topic_terms(topic)
        lexical = bm25(documents, terms)
        ranked, topic_scores = routed_candidates(eligible, forms, topic, vectors, lexical, shortlist)
        lists.append(ranked)
        for identifier, score in topic_scores.items():
            scores[identifier] = max(scores.get(identifier, 0.0), score)
        meaningful = {term for term in route_terms(topic) if " " not in term
                      and term not in {"system", "document", "post"}}
        required = min(2, len(meaningful))
        codes = form_codes(topic)
        best = next((identifier for identifier in ranked if codes
                     or matching_facts(terms, set(documents[identifier])) >= 2
                     or (required and len(meaningful.intersection(documents[identifier])) >= required)), None)
        if best is not None:
            focuses.setdefault(best, []).append(topic)

    # Reserve one candidate per topic before fusing the remainder. A shared
    # passage can be the best candidate for more than one topic.
    remaining = lists[0]
    if len(lists) > 1:
        fused = reciprocal_rank_fusion(*(ranked[:shortlist] for ranked in lists))
        remaining = sorted(fused, key=lambda key: (-fused[key], key))
    ordered = list(dict.fromkeys((*focuses, *remaining)))
    results = []
    for identifier in ordered[:top_k]:
        result = {
            **chunks[identifier], "score": scores[identifier],
            "vector_score": max((vectors[identifier] for vectors in vector_lists
                                 if identifier in vectors), default=None),
            "scope": passage_scope(chunks[identifier]),
        }
        if identifier in focuses:
            result["search_focus"] = focuses[identifier]
        results.append(result)
    covered = {topic for result in results for topic in result.get("search_focus", ())}
    return results, [topic for topic in queries if topic not in covered]


def dense_candidates(target, query, ollama, digest, chunks, kind, limit):
    try:
        vector = ollama.embed(["query: " + query])[0]
    except ContextTooLong:
        raise ValueError("Query exceeds the E5 context limit; shorten the query; truncation is disabled") from None
    if ollama.embedding_digest() != digest:
        raise ValueError("E5 digest changed during search; retry with a stable model")
    candidate_count = min(max(100, limit * 20), len(chunks))
    while True:
        ranking = target.query(query_embeddings=[vector.tolist()],
                               where={"kind": kind} if kind != "both" else None,
                               n_results=candidate_count, include=["distances"])
        scores = {}
        for identifier, distance in zip(ranking["ids"][0], ranking["distances"][0], strict=True):
            if identifier not in chunks:
                raise ValueError("Chroma returned an unexpected passage; index with a new --collection name")
            if not manual_heading_only(chunks[identifier]):
                scores[identifier] = 1.0 - distance
        if len(scores) >= limit or candidate_count == len(chunks):
            return scores
        candidate_count = min(candidate_count * 2, len(chunks))


def search(corpus, collection, query, ollama, *, top_k=3, kind="both",
           chroma_host="127.0.0.1", chroma_port=8000, ranker="legacy", subqueries=(),
           split_query=False):
    started = perf_counter()
    check_kind(kind)
    if ranker not in RANKERS:
        raise ValueError("Unknown ranker; use legacy, dense, bm25, hybrid or routed")
    if subqueries and (ranker != "hybrid" or len(subqueries) > 3):
        raise ValueError("Supply up to three separate searches with --ranker hybrid")
    for phrase in (query, *subqueries):
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("Supply a nonempty search query")
        if len(phrase) > 8192:
            raise ValueError("Search query is too long; shorten it to one focused question")
    if split_query and (ranker != "hybrid" or subqueries):
        raise ValueError("--split-query requires hybrid ranking without explicit subqueries")
    focuses = support_focus(query) if split_query else ()
    queries = (query, *(phrase for _, phrase in focuses)) if focuses else tuple(dict.fromkeys((query, *subqueries)))
    if ranker == "routed":
        queries = query_topics(query)
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
        selected = {identifier: chunk for identifier, chunk in expected.items()
                    if kind == "both" or chunk["kind"] == kind}
        results = []
        priorities = {}
        if selected:
            vector_lists = [
                {} if ranker == "routed" and form_codes(phrase) else
                dense_candidates(target, phrase, ollama, digest, selected, kind,
                                 max(top_k, 20) if ranker == "routed" else top_k)
                for phrase in queries
            ] if ranker != "bm25" else []
            if ranker == "routed":
                results, uncovered = routed_results(selected, queries, vector_lists, top_k=top_k)
                return {"query": query, "kind": kind, "ranker": ranker, "results": results,
                        "search_focuses": list(queries), "uncovered_topics": uncovered,
                        "elapsed_seconds": perf_counter() - started,
                        "corpus_fingerprint": manifest["corpus_fingerprint"], "embedding_digest": digest}
            vector_scores = vector_lists[0] if vector_lists else {}
            exact = {}
            if ranker == "legacy":
                scores = candidate_scores(selected, vector_scores, query, kind)
            elif ranker == "dense":
                scores = vector_scores
            else:
                documents = {identifier: lexical_terms(chunk["embedding_text"].removeprefix("passage: "))
                             for identifier, chunk in selected.items() if not manual_heading_only(chunk)}
                lexical_lists = [bm25(documents, set(lexical_terms(phrase))) for phrase in queries]
                exact = exact_matches({identifier: selected[identifier] for identifier in documents}, query)
                if ranker == "bm25":
                    scores = lexical_lists[0]
                else:
                    shortlist = max(20, top_k)
                    def shortlist_ids(scored):
                        return sorted(scored, key=lambda identifier: (-scored[identifier], identifier))[:shortlist]

                    dense_ranks = [shortlist_ids(scored) for scored in vector_lists]
                    lexical_ranks = [shortlist_ids(scored) for scored in lexical_lists]
                    scores = reciprocal_rank_fusion(*dense_ranks, *lexical_ranks)
                    for (label, phrase), dense, lexical in zip(focuses, dense_ranks[1:], lexical_ranks[1:]):
                        # Require lexical support before reserving a place for a symptom.
                        if not lexical:
                            continue
                        focus_scores = reciprocal_rank_fusion(dense, lexical)
                        focus_exact = exact_matches(
                            {identifier: selected[identifier] for identifier in lexical}, phrase,
                        )
                        best = min(lexical, key=lambda identifier: (
                            -focus_exact.get(identifier, 0), -focus_scores[identifier], identifier,
                        ))
                        priorities.setdefault(best, []).append(label)
                for identifier, matches in exact.items():
                    if matches:
                        scores.setdefault(identifier, 0.0)
            ranked = sorted(scores, key=lambda identifier: (-exact.get(identifier, 0), -scores[identifier], identifier))
            reviewed = reviewed_passages(selected.values()) if focuses else {}
            ordered = list(dict.fromkeys((*reviewed, *priorities, *ranked)))
            for identifier in ordered[:top_k]:
                result = {**selected[identifier], "score": scores.get(identifier, 0.0),
                          "vector_score": vector_scores.get(identifier)}
                if identifier in reviewed:
                    rule, focus = reviewed[identifier]
                    result.update(support_rule=rule, search_focus=[focus])
                elif identifier in priorities:
                    result["search_focus"] = priorities[identifier]
                results.append(result)
        return {"query": query, "kind": kind, "ranker": ranker, "results": results,
                "search_focuses": [label for label, _ in focuses],
                "elapsed_seconds": perf_counter() - started,
                "corpus_fingerprint": manifest["corpus_fingerprint"], "embedding_digest": digest}
