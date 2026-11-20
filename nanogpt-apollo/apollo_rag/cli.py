import argparse
import json
import os
from pathlib import Path
import sys

from .prepare import prepare
from .tokenizer import DEFAULT_CACHE
from .index import build_index
from .chroma import chroma_errors, connect
from .ollama import CHAT_MODEL, DEFAULT_URL, NUM_PREDICT, Ollama
from .retrieval import search
from .answers import ask, print_answer_details
from .evaluation import evaluate, load_cases, sft_examples
from .ranking import RANKERS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Private, traceable Apollo ERP retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="build traceable passages; no indexing or model training")
    command.add_argument("--manual", dest="manuals", action="append", type=Path, default=[], help="UTF-8 manual; repeat for each file")
    command.add_argument("--source-root", type=Path, help="read-only local source checkout")
    command.add_argument("--output", type=Path, required=True, help="new directory outside the source checkout")
    command.add_argument("--tokenizer-cache", type=Path, default=DEFAULT_CACHE)
    command.add_argument("--tokenizer-revision", default="main", help="Hugging Face commit or ref; resolved commit recorded")
    command.add_argument("--offline", action="store_true", help="require tokenizer already present in the local cache")
    command.add_argument("--max-tokens", type=int, default=384)
    command.add_argument("--overlap-tokens", type=int, default=32)
    command.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024)
    command.add_argument("--max-total-bytes", type=int, default=64 * 1024 * 1024)
    command = commands.add_parser("export-sft", help="export reviewed training cases as private chat JSONL")
    command.add_argument("--cases", type=Path, required=True, help="separate reviewed training cases")
    command.add_argument("--holdout", type=Path, required=True, help="evaluation cases to exclude")
    command.add_argument("--corpus", type=Path, required=True)
    command.add_argument("--num-predict", type=int, default=NUM_PREDICT)
    for name, help_text in (
        ("doctor", "check Chroma heartbeat and required Ollama models; never download"),
        ("index", "embed a verified corpus into a resumable Chroma collection"),
        ("search", "show ranked passages and references; no generation"),
        ("ask", "retrieve evidence and stream a cited Serbian Latin answer"),
        ("evaluate", "measure retrieval and optionally generate answers on local benchmark cases"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", DEFAULT_URL))
        command.add_argument("--chroma-host", default=os.environ.get("CHROMA_HOST", "127.0.0.1"))
        command.add_argument("--chroma-port", type=int, default=os.environ.get("CHROMA_PORT", "8000"))
        if name in {"doctor", "ask", "evaluate"}:
            command.add_argument("--model", default=CHAT_MODEL, help="exact local Ollama chat model tag")
        if name in {"ask", "evaluate"}:
            command.add_argument("--num-predict", type=int, default=NUM_PREDICT,
                                 help=f"maximum generated tokens (default {NUM_PREDICT})")
        if name == "doctor":
            continue
        command.add_argument("--corpus", type=Path, required=True)
        command.add_argument("--collection", required=True, help="Chroma collection name; use a new name for changed corpus/model/scope")
        command.add_argument("--kind", choices=("manual", "source", "both"), default="both")
        if name == "index":
            command.add_argument("--batch-size", type=int, default=16)
        else:
            command.add_argument("--ranker", choices=RANKERS, default="legacy")
            command.add_argument("--split-query", action="store_true",
                                 help="prioritize evidence for stock-account and balance symptoms; requires hybrid")
            command.add_argument("--top-k", type=int, default=10 if name == "evaluate" else 3)
            if name == "evaluate":
                command.add_argument("--cases", type=Path, required=True)
                command.add_argument("--case", dest="case_ids", action="append", default=[], help="case ID; repeat to select more")
                command.add_argument("--domain", help="filter by the benchmark case domain")
                command.add_argument("--context-k", type=int, default=3)
                command.add_argument("--answers", action="store_true", help="run Qwen as well as search")
                command.add_argument("--use-focus", action="store_true", help="use the supplied focused queries with hybrid ranking")
                command.add_argument("--gold-context", action="store_true", help="generate using reviewed evidence directly")
                command.add_argument("--gold-notes", action="store_true",
                                     help="use reviewed source notes instead of whole passages; requires --gold-context")
            else:
                command.add_argument("query")
                command.add_argument("--subquery", dest="subqueries", action="append", default=[],
                                     help="focused search to fuse with the main query; requires hybrid")
    args = vars(parser.parse_args(argv))
    operation = args.pop("command")
    try:
        if operation == "prepare":
            manifest = prepare(**args)
            print(f"Prepared {manifest['document_count']} documents, {manifest['chunk_count']} chunks.")
            print(f"Corpus fingerprint: {manifest['corpus_fingerprint']}")
        elif operation == "export-sft":
            cases = load_cases(args.pop("cases"))
            holdout = load_cases(args.pop("holdout"))
            for row in sft_examples(cases, holdout, **args):
                print(json.dumps(row, ensure_ascii=False))
        else:
            endpoint = args.pop("ollama_url")
            chat_model = args.pop("model", CHAT_MODEL)
            num_predict = args.pop("num_predict", NUM_PREDICT)
            with Ollama(endpoint, chat_model=chat_model, num_predict=num_predict) as ollama:
                if operation == "doctor":
                    with chroma_errors():
                        heartbeat = connect(args["chroma_host"], args["chroma_port"]).heartbeat()
                    print(f"Chroma heartbeat: {heartbeat}.")
                    status = ollama.doctor()
                    print(f"Ollama {status['version']} is ready.")
                    for name, digest in status["models"].items():
                        print(f"{name}: {digest}")
                elif operation == "index":
                    def progress(done, total):
                        print(f"Indexed or reused {done}/{total} chunks.", file=sys.stderr, flush=True)

                    status = build_index(ollama=ollama, progress=progress, **args)
                    print(f"Indexed {status['chunk_count']} {status['kind']} chunks; {status['reused']} reused; {status['embedding_requests']} embedding requests.")
                    print(f"Corpus fingerprint: {status['corpus_fingerprint']}")
                elif operation == "evaluate":
                    cases = load_cases(args.pop("cases"))
                    case_ids = set(args.pop("case_ids"))
                    domain = args.pop("domain")
                    if not case_ids <= {case["id"] for case in cases}:
                        raise ValueError("Unknown benchmark case ID; see the cases file")
                    cases = [case for case in cases if (not case_ids or case["id"] in case_ids)
                             and (domain is None or case["domain"] == domain)]
                    if not cases:
                        raise ValueError("No benchmark cases match those filters")
                    def progress(number, total, case_id):
                        print(f"Evaluating {number}/{total} {case_id}", file=sys.stderr, flush=True)

                    report = evaluate(cases, ollama=ollama, progress=progress, **args)
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                elif operation == "ask":
                    if chat_model not in ollama.models():
                        raise ValueError("Chat model is missing or remote; use a local tag from ollama list")
                    status = ask(ollama=ollama, on_text=lambda text: print(text, end="", flush=True), **args)
                    print_answer_details(status)
                    if status["error"]:
                        raise RuntimeError(status["error"])
                else:
                    status = search(ollama=ollama, **args)
                    for number, result in enumerate(status["results"], 1):
                        pages = (f"pages {result['page_start']}-{result['page_end']} "
                                 if result["kind"] == "manual" else "")
                        cosine = f"{result['vector_score']:.6f}" if result['vector_score'] is not None else "n/a"
                        print(f"[{number}] {result['kind']} {result['path']}:{result['line_start']}-{result['line_end']} "
                              f"{pages}score={result['score']:.6f} cosine={cosine} id={result['id']}")
                        if result.get("scope"):
                            print("    " + " | ".join(filter(None, (result.get("section"), result["scope"]))))
                        print(result["text"])
                        print()
                    if status.get("uncovered_topics"):
                        print("Teme bez ciljanog dokaza: " + " | ".join(status["uncovered_topics"]))
                    print(f"Retrieved {len(status['results'])} passages in {status['elapsed_seconds']:.3f}s.")
    except (OSError, ValueError, RuntimeError):
        # Known diagnostics never echo server bodies, credentials, or file text.
        error = sys.exc_info()[1]
        if isinstance(error, (ValueError, RuntimeError)):
            print(f"{operation} failed: {error}", file=sys.stderr)
        else:
            print(f"{operation} failed: a local file operation failed; check input/output permissions", file=sys.stderr)
        return 1
    return 0
