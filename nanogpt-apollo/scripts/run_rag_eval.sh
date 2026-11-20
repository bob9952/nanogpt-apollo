#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_rag_eval.sh retrieval [evaluate options]
  bash scripts/run_rag_eval.sh answers [evaluate options]
  bash scripts/run_rag_eval.sh routing [evaluate options]
  bash scripts/run_rag_eval.sh routing-answers [evaluate options]
  bash scripts/run_rag_eval.sh run VERSION LABEL [evaluate options]

retrieval compares v5 legacy, v6 legacy and v6 hybrid.
answers compares legacy, hybrid, focused hybrid and a fixed-evidence ITEM-56 control.
routing compares hybrid and routed evidence selection on the same target corpus.
routing-answers compares Qwen answers across seven selected support questions.
run executes one custom evaluation, for example run v6 qwen-2b --answers --model TAG.
Use APOLLO_EVAL_BASELINE=v5 and APOLLO_EVAL_TARGET=v6 to select the matrix corpora.
Use APOLLO_EVAL_CASES to set the case file and APOLLO_EVAL_OUTPUT to set the output parent.
Set APOLLO_EVAL_BUILD=1 to rebuild the assistant once before a run.
EOF
}

mode=${1:-}
case "$mode" in
  -h|--help) usage; exit 0 ;;
  retrieval|answers|routing|routing-answers|run) shift ;;
  *) usage >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
baseline=${APOLLO_EVAL_BASELINE:-v5}
target=${APOLLO_EVAL_TARGET:-v6}
case_file=${APOLLO_EVAL_CASES:-"$PWD/docs/serbian-support-cases.json"}
output=${APOLLO_EVAL_OUTPUT:-"$PWD/data/evaluation"}
report_paths=()

if [[ "$mode" == run ]]; then
  if (( $# < 2 )); then
    usage >&2
    exit 2
  fi
  target=$1
  label=$2
  shift 2
  if [[ ! "$label" =~ ^[[:alnum:]_-]+$ ]]; then
    echo "Use letters, digits, underscores and hyphens for the report label." >&2
    exit 2
  fi
fi

# Matrix runs set these options themselves. Use run for a custom combination.
for arg in "$@"; do
  case "${arg%%=*}" in
    --corpus|--collection|--cases|--kind)
      echo "Set the corpus version and case file through the runner. Scope is manual." >&2
      exit 2 ;;
    --ranker|--case|--domain|--answers|--use-focus|--gold-context|--gold-notes|--split-query)
      if [[ "$mode" != run ]]; then
        echo "Use the run mode to customize $arg." >&2
        exit 2
      fi ;;
  esac
done

check_corpus() {
  local version=$1
  if [[ ! "$version" =~ ^v[0-9]+$ ]] || [[ ! -d "$PWD/data/private/corpus-$version" ]]; then
    echo "Corpus directory missing or invalid version: $PWD/data/private/corpus-$version" >&2
    exit 2
  fi
}

check_corpus "$target"
if [[ "$mode" == retrieval ]]; then
  check_corpus "$baseline"
fi
if [[ ! -f "$case_file" || ! -r "$case_file" ]]; then
  echo "Case file is missing or unreadable: $case_file" >&2
  exit 2
fi
case_file=$(realpath -- "$case_file")

umask 077
mkdir -p -- "$output"
run_dir=$(mktemp -d -- "$output/$(date -u +%Y%m%dT%H%M%SZ)-$mode-XXXXXX")
printf 'Reports: %s\n' "$run_dir"
if [[ "${APOLLO_EVAL_BUILD:-0}" == 1 ]]; then
  docker compose build assistant
fi

run_eval() {
  local version=$1 name=$2
  shift 2
  local report="$run_dir/$name.json"
  printf '\nRunning %s on %s\n' "$name" "$version" >&2
  if APOLLO_CORPUS_DIR="$PWD/data/private/corpus-$version" docker compose run --rm -T \
    -v "$case_file:/cases.json:ro" \
    assistant evaluate --cases /cases.json --corpus /corpus \
    --collection "apollo-manuals-$version" --kind manual "$@" > "$report.partial"; then
    mv -- "$report.partial" "$report"
  else
    echo "Evaluation failed. See $report.partial and the console output." >&2
    return 1
  fi
  report_paths+=("$report")
  printf 'Saved %s\n' "$report" >&2
}

case "$mode" in
  retrieval)
    run_eval "$baseline" "$baseline-legacy" --ranker legacy "$@"
    if [[ "$target" != "$baseline" ]]; then
      run_eval "$target" "$target-legacy" --ranker legacy "$@"
    fi
    run_eval "$target" "$target-hybrid" --ranker hybrid "$@"
    ;;
  routing)
    run_eval "$target" "$target-hybrid" --ranker hybrid "$@"
    run_eval "$target" "$target-routed" --ranker routed "$@"
    ;;
  routing-answers)
    smoke=(--case ITEM-56 --case STOCK-DEFAULTS --case FDC-EXACT --case Q03 --case Q21 --case LIVE-ACCOUNT --case OUT-OF-SCOPE)
    run_eval "$target" "$target-hybrid-answers" --ranker hybrid --answers "${smoke[@]}" "$@"
    run_eval "$target" "$target-routed-answers" --ranker routed --answers "${smoke[@]}" "$@"
    ;;
  answers)
    smoke=(--case ITEM-56 --case STOCK-DEFAULTS --case LIVE-ACCOUNT --case OUT-OF-SCOPE)
    run_eval "$target" legacy --ranker legacy --answers "${smoke[@]}" "$@"
    run_eval "$target" hybrid --ranker hybrid --answers "${smoke[@]}" "$@"
    run_eval "$target" hybrid-focused --ranker hybrid --answers "${smoke[@]}" --use-focus "$@"
    run_eval "$target" item56-fixed-evidence --ranker hybrid --answers --gold-context --case ITEM-56 "$@"
    ;;
  run)
    run_eval "$target" "$label" "$@"
    ;;
esac

summary_args=()
if [[ "$mode" == routing-answers ]]; then
  summary_args=(--case ITEM-56)
fi
python3 scripts/summarize_rag_eval.py "${summary_args[@]}" "${report_paths[@]}" | tee "$run_dir/summary.txt"
