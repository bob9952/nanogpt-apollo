"""Summarize private Apollo evaluation reports using the Python standard library."""

import argparse
import json
from pathlib import Path
from statistics import median


def load_report(path):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        for key in ("summary", "cases", "collection", "ranker"):
            report[key]
        return report
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Cannot read evaluation report {path}: {error}") from None


def show_summary(path, report):
    summary = report["summary"]
    mode = report["ranker"]
    if report.get("gold_notes"):
        mode += " / reviewed source notes"
    elif report.get("gold_context"):
        mode += " / fixed evidence"
    if report.get("use_focus"):
        mode += " / focused"
    indexed = summary["fully_indexed_annotated_cases"]
    print(f"\n{path.stem} | {report['collection']} | {mode}")
    print(f"  Cases {summary['cases']}, with source anchors {summary['annotated_cases']}, "
          f"fully indexed {indexed}")
    print(f"  Cases with a top-5 anchor {summary['annotated_cases_with_hit_in_first_5']}")
    answers = [case["answer"] for case in report["cases"] if case["answer"] is not None]
    reviewed = sum(answer.get("method") == "reviewed_support" for answer in answers)
    prompt = "fixed prompt" if report.get("gold_context") else "prompt"
    if reviewed:
        prompt = "answer evidence"
    print(f"  All anchors ranked {summary['all_anchors_ranked_for_fully_indexed_cases']}/{indexed}, "
          f"all anchors in {prompt} {summary['all_anchors_in_context_for_fully_indexed_cases']}/{indexed}")
    print(f"  Mean retrieval {summary['mean_retrieval_seconds']:.2f}s")
    print(f"  Corpus {report.get('corpus_fingerprint', 'unknown')}, "
          f"embedder {report.get('embedding_digest', 'unknown')}")
    if report.get("case_set_sha256"):
        print(f"  Case set {report['case_set_sha256']}")
    if report.get("answer_system_sha256"):
        print(f"  System prompt {report['answer_system_sha256']}")
    if "num_predict" in report:
        print(f"  Output token limit {report['num_predict']}")
    if not report.get("chat_model"):
        return
    incomplete = sum(not answer["complete"] for answer in answers)
    truncated = sum(answer.get("done_reason") == "length" for answer in answers)
    warnings = sum(bool(answer.get("warnings")) for answer in answers)
    if reviewed:
        print(f"  Reviewed support answers {reviewed}")
    if reviewed < len(answers) or not answers:
        print(f"  {report['chat_model']} ({report.get('chat_digest') or 'no digest'})")
    print(f"  Answers with errors {summary['answer_errors']}, incomplete {incomplete}, "
          f"token limit {truncated}, with warnings {warnings}")
    for label, key in (("first text", "first_text_seconds"), ("total", "total_seconds"),
                       ("model load", "load_seconds")):
        values = [answer[key] for answer in answers if answer.get(key) is not None]
        if values:
            print(f"  Median {label} {median(values):.2f}s")


def show_comparison(baseline_path, baseline, path, report):
    if (baseline.get("gold_context") or report.get("gold_context")
            or baseline.get("embedding_digest") != report.get("embedding_digest")
            or (baseline.get("top_k"), baseline.get("context_k"), baseline.get("num_predict", 256))
            != (report.get("top_k"), report.get("context_k"), report.get("num_predict", 256))):
        return
    previous = {case["id"]: case for case in baseline["cases"]}
    paired = []
    for case in report["cases"]:
        old = previous.get(case["id"])
        expected = case["expected_evidence"]
        if (old and expected and old["query"] == case["query"]
                and old["expected_evidence"] == expected
                and old.get("evidence_alternatives") == case.get("evidence_alternatives")
                and old["indexed_evidence_spans"] == case["indexed_evidence_spans"] == len(expected)):
            paired.append((old, case))
    if not paired:
        return
    improved = [case["id"] for old, case in paired
                if case["matched_context_spans"] > old["matched_context_spans"]]
    regressed = [case["id"] for old, case in paired
                 if case["matched_context_spans"] < old["matched_context_spans"]]
    print(f"\n{baseline_path.stem} → {path.stem} | {len(paired)} shared fully indexed cases")
    print(f"  In-prompt anchor count improved: {', '.join(improved) or 'none'}")
    print(f"  In-prompt anchor count regressed: {', '.join(regressed) or 'none'}")


def show_case(case):
    expected = case["expected_evidence"]
    answer = case["answer"]
    context_label = "in answer evidence" if (answer or {}).get("method") == "reviewed_support" else "in prompt"
    print(f"  {case['id']} | {case['query']}")
    if answer is not None:
        label = {"reviewed_support": "Reviewed support answer",
                 "insufficient_evidence": "Evidence gap"}.get(answer.get("method"), "Generated answer")
        print(f"  {label}: {answer['text'] or '(empty)'}")
        for warning in answer.get("warnings", []):
            print(f"  Warning: {warning}")
        if answer.get("error"):
            print(f"  Error: {answer['error']}")
    if case.get("uncovered_topics"):
        print(f"  Topics without targeted evidence: {' | '.join(case['uncovered_topics'])}")
    if expected:
        print(f"  Anchors indexed {case['indexed_evidence_spans']}/{len(expected)}, "
              f"ranked {case['matched_ranked_spans']}/{len(expected)}, "
              f"{context_label} {case['matched_context_spans']}/{len(expected)}")
        alternatives = case.get("evidence_alternatives") or [[] for _ in expected]
        for (source, first, last), equivalents in zip(expected, alternatives):
            locations = [f"{path}:{start}-{end}" for path, start, end in equivalents]
            print(f"    Expected {source}:{first}-{last}" + (f" or {', '.join(locations)}" if locations else ""))
        if case["missing_documents"]:
            print(f"    Missing documents: {', '.join(case['missing_documents'])}")
    else:
        print("  No reviewed source anchors. Grade against the review notes.")
    for passage in case["context"]:
        print(f"    [{passage['citation']}] {passage['path']}:{passage['line_start']}-{passage['line_end']}")
        if passage.get("scope"):
            print("      " + " | ".join(filter(None, (passage.get("section"), passage["scope"]))))
    print(f"  Review: {case['review']}")
    if case.get("reference_answer"):
        print(f"  Reference: {case['reference_answer']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="JSON reports or a run directory")
    parser.add_argument("--case", action="append", default=[], help="show answers and evidence for a case ID")
    args = parser.parse_args()
    paths = []
    for path in args.reports:
        paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    if not paths:
        parser.error("No completed JSON reports found")
    reports = [(path, load_report(path)) for path in paths]
    if len(args.reports) == 1 and args.reports[0].is_dir():
        reports.sort(key=lambda entry: entry[1]["ranker"] != "legacy")
    case_ids = set(args.case)
    if case_ids:
        unknown = case_ids - {case["id"] for _, report in reports for case in report["cases"]}
        if unknown:
            parser.error("Unknown case IDs: " + ", ".join(sorted(unknown)))
    print("Source-anchor counts measure line overlap. Review answer correctness and citation support manually.")
    for path, report in reports:
        show_summary(path, report)
        single_case_run = not case_ids and len(report["cases"]) == 1
        for case in report["cases"]:
            if case["id"] in case_ids or (single_case_run and case["answer"] is not None):
                show_case(case)
    legacy = {report["collection"]: (path, report) for path, report in reports
              if report["ranker"] == "legacy" and not report.get("gold_context")}
    for path, report in reports[1:]:
        show_comparison(*reports[0], path, report)
        same_collection = legacy.get(report["collection"])
        if same_collection and same_collection[0] not in (path, reports[0][0]):
            show_comparison(*same_collection, path, report)


if __name__ == "__main__":
    main()
