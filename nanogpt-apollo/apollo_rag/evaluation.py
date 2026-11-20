"""Evaluate ranked evidence and answers using a local, source-anchored question set."""

from hashlib import sha256
import json
import re
from statistics import mean
from time import perf_counter

from .answers import SYSTEM, answer_from_results, answer_prompt, answer_system, select_passages
from .bundle import load_bundle
from .ollama import NUM_CTX
from .retrieval import manual_heading_only, normalized_words, search


def load_cases(path):
    try:
        cases = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(cases, list) or not cases:
            raise ValueError
        identifiers = set()
        for case in cases:
            identifier = case["id"]
            if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Z0-9-]+", identifier)
                    or identifier in identifiers or not case["query"].strip()
                    or not case["domain"].strip() or not case["review"].strip()):
                raise ValueError
            identifiers.add(identifier)
            evidence = case.get("evidence", [])
            if not isinstance(evidence, list):
                raise ValueError
            alternatives = case.get("evidence_alternatives", [])
            if not isinstance(alternatives, list) or alternatives and (
                len(alternatives) != len(evidence) or not all(isinstance(group, list) for group in alternatives)
            ):
                raise ValueError
            for path, start, end in [*evidence, *(span for group in alternatives for span in group)]:
                if (not isinstance(path, str) or not path or type(start) is not int
                        or type(end) is not int or not 1 <= start <= end):
                    raise ValueError
            notes = case.get("evidence_notes")
            if notes is not None and (
                not isinstance(notes, list) or not evidence or len(notes) != len(evidence)
                or not all(isinstance(note, str) and note.strip() for note in notes)
            ):
                raise ValueError
            focus = case.get("focus", [])
            if not isinstance(focus, list) or len(focus) > 3 or not all(
                isinstance(query, str) and query.strip() for query in focus
            ):
                raise ValueError
            reference = case.get("reference_answer")
            if reference is not None and (not isinstance(reference, str) or not reference.strip()):
                raise ValueError
    except (AttributeError, KeyError, OSError, TypeError, UnicodeError, ValueError):
        raise ValueError("Invalid benchmark cases file; use the checked-in Serbian cases as the template") from None
    return cases


def overlap(span, passage):
    path, first, last = span
    if passage["path"] != path:
        return 0
    return max(0, min(last, passage["line_end"]) - max(first, passage["line_start"]) + 1)


def matching_spans(spans, passages, alternatives=()):
    groups = alternatives or [[] for _ in spans]
    return [span for span, equivalent in zip(spans, groups, strict=True)
            if any(overlap(candidate, passage) for candidate in (span, *equivalent) for passage in passages)]


def gold_passages(spans, chunks, notes=None):
    """Select corpus passages or use reviewed notes for the same source lines."""
    ranked = []
    seen = set()
    for index, span in enumerate(spans):
        candidates = [chunk for chunk in chunks if overlap(span, chunk) and not manual_heading_only(chunk)]
        if not candidates:
            continue
        winner = min(candidates, key=lambda chunk: (-overlap(span, chunk), chunk["token_count"], chunk["id"]))
        if notes is not None:
            note = notes[index]
            first, last = span[1:]
            if overlap(span, winner) != last - first + 1:
                continue
            winner = {
                **winner,
                "id": sha256(json.dumps([winner["id"], span, note]).encode("utf-8")).hexdigest(),
                "original_id": winner["id"],
                "line_start": first, "line_end": last, "text": note,
            }
        if winner["id"] not in seen:
            ranked.append(winner)
            seen.add(winner["id"])
    return ranked


def sft_examples(cases, holdout, corpus, *, num_predict):
    """Build chat records from separately reviewed, source-anchored answers."""
    if not 1 <= num_predict < NUM_CTX:
        raise ValueError(f"--num-predict must be between 1 and {NUM_CTX - 1} tokens")
    evaluation_ids = {case["id"] for case in holdout}
    questions = {tuple(normalized_words(case["query"])) for case in holdout}
    _, chunks = load_bundle(corpus)
    rows = []
    for case in cases:
        identifier = case["id"]
        question = tuple(normalized_words(case["query"]))
        if not question or identifier in evaluation_ids or question in questions:
            raise ValueError(f"{identifier}: empty, duplicate, or held-out training question")
        questions.add(question)
        spans = case.get("evidence")
        answer = case.get("reference_answer")
        if not spans or not answer:
            raise ValueError(f"{identifier}: training requires reviewed evidence and a cited reference answer")
        passages = gold_passages(spans, chunks)
        if not passages:
            raise ValueError(f"{identifier}: no reviewed evidence was found in the corpus")
        sources = select_passages(case["query"], passages, num_predict=num_predict)
        if any(not any(overlap(span, source) == span[2] - span[1] + 1 for source in sources)
               for span in spans):
            raise ValueError(f"{identifier}: reviewed evidence is missing or does not fit the training prompt")
        cited = set(re.findall(r"\[(P\d+)\]", answer))
        if not cited or not cited <= {source["citation"] for source in sources}:
            raise ValueError(f"{identifier}: use citations from the reviewed passages in evidence order")
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": answer_prompt(case["query"], sources)},
            {"role": "assistant", "content": answer},
        ]})
    return rows


def passage_record(passage):
    keys = ("id", "path", "line_start", "line_end", "page_start", "page_end", "section",
            "score", "vector_score", "citation", "text", "original_id", "search_focus", "excerpt", "support_rule", "scope")
    return {key: passage[key] for key in keys if key in passage}


def evaluate(cases, corpus, collection, ollama, *, ranker, kind, top_k,
             context_k, answers, use_focus, gold_context, chroma_host, chroma_port, progress=None,
             gold_notes=False, split_query=False):
    if not cases:
        raise ValueError("No benchmark cases selected")
    if not 1 <= context_k <= top_k <= 100:
        raise ValueError("Require 1 <= --context-k <= --top-k <= 100")
    if use_focus and ranker != "hybrid":
        raise ValueError("--use-focus requires --ranker hybrid")
    if split_query and (use_focus or gold_context or ranker != "hybrid"):
        raise ValueError("--split-query requires hybrid ranking without --use-focus or --gold-context")
    routed = ranker == "routed"
    if routed and gold_context:
        raise ValueError("Use the hybrid ranker for fixed-evidence controls")
    if gold_context and (not answers or any(not case.get("evidence") for case in cases)):
        raise ValueError("--gold-context requires --answers and cases with reviewed evidence; choose --case IDs")
    if gold_notes and (not gold_context or any(not case.get("evidence_notes") for case in cases)):
        raise ValueError("--gold-notes requires --gold-context and reviewed notes for each selected case")
    manifest, chunks = load_bundle(corpus)
    chunks = [chunk for chunk in chunks if (kind == "both" or chunk["kind"] == kind)
              and not manual_heading_only(chunk)]
    paths = {chunk["path"] for chunk in chunks}
    controls = {}
    if gold_context:
        for case in cases:
            spans = case["evidence"]
            notes = case["evidence_notes"] if gold_notes else None
            controls[case["id"]] = gold_passages(spans, chunks, notes)
            if len(matching_spans(spans, controls[case["id"]])) != len(spans):
                raise ValueError(f"Case {case['id']} lacks its reviewed evidence in this corpus")

    models = ollama.models() if answers else {}
    if answers and ollama.chat_model not in models:
        raise ValueError("Chat model is missing or remote; choose a tag from ollama list")
    records = []
    ragchecker_rows = []
    for number, case in enumerate(cases, 1):
        if progress is not None:
            progress(number, len(cases), case["id"])
        started = perf_counter()
        query = case["query"]
        result = search(corpus, collection, query, ollama, kind=kind, ranker=ranker,
                        top_k=top_k, subqueries=case.get("focus", []) if use_focus else (),
                        chroma_host=chroma_host, chroma_port=chroma_port, split_query=split_query)
        if result["corpus_fingerprint"] != manifest["corpus_fingerprint"]:
            raise ValueError("Corpus changed during evaluation; rerun against a stable bundle")
        if records and result["embedding_digest"] != embedding_digest:
            raise ValueError("Embedding model changed during evaluation; rerun against a stable digest")
        embedding_digest = result["embedding_digest"]
        ranked = result["results"]
        evidence = case.get("evidence", [])
        alternatives = case.get("evidence_alternatives", [])
        covered = matching_spans(evidence, chunks, alternatives)
        groups = alternatives or [[] for _ in evidence]
        missing = sorted({span[0] for span, equivalent in zip(evidence, groups, strict=True)
                          if not any(candidate[0] in paths for candidate in (span, *equivalent))})
        best = controls.get(case["id"], ranked[:context_k])
        answer = None
        context = []
        topics = result.get("search_focuses", ())
        if answers and (best or routed):
            answer = answer_from_results(query, best, ollama,
                                         retrieval_seconds=result["elapsed_seconds"], started=started,
                                         search_focuses=topics, routed=routed)
            context = answer.pop("sources")
        elif best:
            context = select_passages(query, best, num_predict=ollama.num_predict, topics=topics if routed else ())
        elif answers:
            answer = {"complete": False, "error": "No passages found", "text": ""}
        found = matching_spans(evidence, ranked, alternatives)
        used = matching_spans(evidence, context, alternatives)
        context_topics = {topic for passage in context for topic in passage.get("search_focus", ())}
        records.append({
            "id": case["id"], "query": query, "domain": case["domain"], "review": case["review"],
            "reference_answer": case.get("reference_answer"),
            "expected_evidence": evidence, "missing_documents": missing,
            **({"evidence_alternatives": alternatives} if alternatives else {}),
            "indexed_evidence_spans": len(covered),
            "matched_first_5_spans": len(matching_spans(evidence, ranked[:5], alternatives)),
            "matched_ranked_spans": len(found), "matched_context_spans": len(used),
            "retrieval_seconds": result["elapsed_seconds"], "search_focuses": topics,
            **({"uncovered_topics": [topic for topic in topics if topic not in context_topics]} if routed else {}),
            "ranked": [passage_record(passage) for passage in ranked],
            "context": [passage_record(passage) for passage in context], "answer": answer,
        })
        if (answer and answer["complete"] and answer.get("done_reason") != "length"
                and case.get("reference_answer")
                and (not evidence or len(covered) == len(evidence))):
            ragchecker_rows.append({
                "query_id": case["id"], "query": query, "gt_answer": case["reference_answer"],
                "response": answer["text"],
                "retrieved_context": [{"doc_id": passage["id"], "text": passage["text"]} for passage in context],
            })

    annotated = [r for r in records if r["expected_evidence"]]
    some_indexed = [r for r in annotated if r["indexed_evidence_spans"]]
    fully_indexed = [r for r in annotated if r["indexed_evidence_spans"] == len(r["expected_evidence"])]
    if answers and ollama.models().get(ollama.chat_model) != models[ollama.chat_model]:
        raise ValueError("Chat model changed during evaluation; rerun against a stable digest")
    return {
        "collection": collection, "corpus_fingerprint": manifest["corpus_fingerprint"],
        "embedding_digest": embedding_digest, "ranker": ranker, "top_k": top_k, "context_k": context_k,
        "kind": kind, "use_focus": use_focus, "gold_context": gold_context, "split_query": split_query,
        "num_predict": ollama.num_predict, "gold_notes": gold_notes,
        "case_set_sha256": sha256(json.dumps(cases, sort_keys=True).encode("utf-8")).hexdigest(),
        "answer_system_sha256": sha256(answer_system(routed).encode("utf-8")).hexdigest(),
        "chat_model": ollama.chat_model if answers else None,
        "chat_digest": models.get(ollama.chat_model),
        "summary": {
            "cases": len(records), "annotated_cases": len(annotated),
            "fully_indexed_annotated_cases": len(fully_indexed),
            "annotated_cases_with_hit_in_first_5": sum(r["matched_first_5_spans"] > 0 for r in annotated),
            "mean_ranked_span_recall": (
                mean(r["matched_ranked_spans"] / len(r["expected_evidence"]) for r in annotated)
                if annotated else None
            ),
            "mean_ranked_recall_for_indexed_spans": (
                mean(r["matched_ranked_spans"] / r["indexed_evidence_spans"] for r in some_indexed)
                if some_indexed else None
            ),
            "all_anchors_ranked_for_fully_indexed_cases": sum(
                r["matched_ranked_spans"] == len(r["expected_evidence"]) for r in fully_indexed
            ),
            "all_anchors_in_context_for_fully_indexed_cases": sum(
                r["matched_context_spans"] == len(r["expected_evidence"]) for r in fully_indexed
            ),
            "answer_errors": sum(bool(r["answer"] and r["answer"]["error"]) for r in records),
            "mean_retrieval_seconds": mean(r["retrieval_seconds"] for r in records),
        },
        "cases": records, "ragchecker": {"results": ragchecker_rows},
    }
