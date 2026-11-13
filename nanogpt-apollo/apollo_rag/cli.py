import argparse
import os
from pathlib import Path
import sys

from .prepare import prepare
from .tokenizer import DEFAULT_CACHE
from .index import build_index
from .chroma import chroma_errors, connect
from .ollama import DEFAULT_URL, Ollama
from .retrieval import search
from .answers import ask, print_answer_details


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
    for name, help_text in (
        ("doctor", "check Chroma heartbeat and required Ollama models; never download"),
        ("index", "embed a verified corpus into a resumable Chroma collection"),
        ("search", "show ranked passages and references; no generation"),
        ("ask", "retrieve evidence and stream a cited Serbian Latin answer"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", DEFAULT_URL))
        command.add_argument("--chroma-host", default=os.environ.get("CHROMA_HOST", "127.0.0.1"))
        command.add_argument("--chroma-port", type=int, default=os.environ.get("CHROMA_PORT", "8000"))
        if name == "doctor":
            continue
        command.add_argument("--corpus", type=Path, required=True)
        command.add_argument("--collection", required=True, help="Chroma collection name; use a new name for changed corpus/model/scope")
        command.add_argument("--kind", choices=("manual", "source", "both"), default="both")
        if name == "index":
            command.add_argument("--batch-size", type=int, default=16)
        else:
            command.add_argument("query")
            command.add_argument("--top-k", type=int, default=3)
    args = vars(parser.parse_args(argv))
    operation = args.pop("command")
    try:
        if operation == "prepare":
            manifest = prepare(**args)
            print(f"Prepared {manifest['document_count']} documents, {manifest['chunk_count']} chunks.")
            print(f"Corpus fingerprint: {manifest['corpus_fingerprint']}")
        else:
            endpoint = args.pop("ollama_url")
            with Ollama(endpoint) as ollama:
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
                elif operation == "ask":
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
                        print(result["text"])
                        print()
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
