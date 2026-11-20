# Serbian Apollo RAG evaluation

This benchmark uses the indexed Apollo manuals and the checked-in support questions.
The reports contain private passages. Keep them under the ignored `data/evaluation/` directory.
[Question set](serbian-support-cases.json)

## Run on Ubuntu

Run these from the app checkout in PuTTY. The runner reuses the existing Chroma collections and corpus bundles. It writes each run to a new folder under the ignored `data/evaluation/`. Per-case progress stays on the console, complete results go into JSON, and `summary.txt` gives the comparison. A failed command leaves its `.partial` file and preserves completed reports.

```bash
cd ~/ai/nanogpt-apollo/nanogpt-apollo
APOLLO_EVAL_BUILD=1 bash scripts/run_rag_eval.sh retrieval
bash scripts/run_rag_eval.sh answers
```

`retrieval` runs 35 questions per configuration with v5 legacy, v6 legacy and v6 hybrid. The v5 and v6 legacy pair isolates the corpus change. The v6 pair isolates ranking. `answers` runs four support controls with legacy, hybrid and focused hybrid, plus one fixed-evidence ITEM-56 run. The focused case uses prewritten queries and measures what those queries add. The fixed-evidence case uses reviewed passages directly and still applies the prompt budget.

Use `run` to change one setting, try another model, or run all questions with generation. Pass any extra evaluation options after the report label. `APOLLO_EVAL_BASELINE` and `APOLLO_EVAL_TARGET` change the preset corpus versions. `APOLLO_EVAL_CASES` selects a different JSON case file.

```bash
bash scripts/run_rag_eval.sh run v6 item56-fixed --answers --gold-context --case ITEM-56 --ranker hybrid

mkdir -p data/evaluation
umask 077
nohup bash scripts/run_rag_eval.sh run v6 hybrid-full --answers --ranker hybrid \
  > "data/evaluation/hybrid-full-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 </dev/null &
```

Use `--num-predict 512` to test a higher output limit with `run` or with the answer preset. The default is 256. The limit reserves space inside the same 4,096-token context. Check the actual `context` in both reports when comparing answers.

```bash
bash scripts/run_rag_eval.sh run v6 item56-fixed-9b-512 \
  --ranker hybrid --answers --gold-context --case ITEM-56 \
  --model qwen3.5:9b-q4_K_M --num-predict 512
```

Use the report directory printed by the runner, or pass specific reports in baseline-first order.

```bash
python3 scripts/summarize_rag_eval.py --case ITEM-56 data/evaluation/RUN_DIRECTORY
python3 scripts/summarize_rag_eval.py --case ITEM-56 \
  data/evaluation/item56-fixed-evidence.json path/to/new/item56-fixed.json
```

## Compare routed retrieval with the hybrid baseline

Rebuild once and run the same 35 questions on the v6 index. The first command
compares evidence selection without generation. The second generates Qwen
answers for a smaller set and prints ITEM-56 in the console for both runs.

```bash
APOLLO_EVAL_BUILD=1 bash scripts/run_rag_eval.sh routing
bash scripts/run_rag_eval.sh routing-answers
```

Use `--ranker routed` on `search`, `ask`, or a custom evaluation. It routes
explicit form codes to exact matches and uses E5 plus BM25 for free-text topics.
The route splits distinct requests, ranks the passages for each, reserves
evidence per topic, and keeps the source heading and its known scope in the
prompt. It reports uncovered topics after fitting the actual prompt. It uses
the existing corpus and index. The reviewed checklist is a separate control.

Compare fully indexed cases first. Inspect lost and gained anchors and manually
grade both answer correctness and citations. ITEM-56 accepts the G4
manual's stock-account fallback at lines 1272–1278 as an equivalent source
for the same rule in the installation manual. Check that the actual excerpt
contains the lookup conditions. The routed generation path also
uses a shorter, scope-aware prompt, so treat the answer matrix as a comparison
of the full pipelines. Keep the same model, corpus, and token limit. If Qwen
misses facts repeatedly on fresh cases with verified passages in the prompt,
use the fixed-evidence control and reviewed SFT export before spending on
training. Keep private evidence local unless an upload is authorized.

## Test evidence for both ITEM-56 symptoms

Use the same model, question, system prompt and output limit as the earlier
notes run. The notes cover the existing stock-record lookup, the G4 account
defaults, and the separate G4 description of financial imbalance. Use the
configuration-dependent stock rule conditionally for a general Apollo question.
The app records the notes with their source lines and original passage IDs.
The reference answer stays in the evaluation report.

```bash
APOLLO_EVAL_BUILD=1 bash scripts/run_rag_eval.sh run v6 item56-4b-balance-notes-512 \
  --ranker hybrid --answers --gold-context --gold-notes --case ITEM-56 \
  --model qwen3.5:4b-q4_K_M --num-predict 512
```

Review the lookup keys, unsupported causal claims, and follow-up on the
imbalance. This check isolates the evidence supplied to the model. The
production retrieval path still needs its own evaluation.

## Search for both symptoms

Split search prioritizes passages for a stock-account problem and a document
imbalance. When the full passages for both topics exceed the prompt budget,
the answer uses verbatim excerpts with the original source line ranges. The
report records `excerpt: true` for a shortened passage in `context`.
Inspect those ranges before adding `--answers` to the same command.

```bash
APOLLO_EVAL_BUILD=1 bash scripts/run_rag_eval.sh run v6 item56-split-retrieval \
  --ranker hybrid --split-query --case ITEM-56 --num-predict 512
```

The same `--ranker hybrid --split-query` options work with `ask`.
This uses the existing Chroma collection.

## Reviewed answer for stock account and imbalance

With `--ranker hybrid --split-query`, Apollo recognizes the symptom pair
and looks for the documented rules in the selected corpus. When both G4 rules
are found, it prints cited checks from reviewed guidance. It adds the
installation-specific posting rule only when its passage is available and
states it conditionally. The report labels the result as a reviewed support
answer. Qwen handles questions without the verified pair.

```bash
APOLLO_EVAL_BUILD=1 bash scripts/run_rag_eval.sh run v6 item56-reviewed \
  --ranker hybrid --split-query --case ITEM-56 --answers
```

## Export reviewed examples for SFT

Use the benchmark case schema to write fresh, reviewed training questions in
`data/private/sft-train.json`. Give them distinct IDs, source ranges, and a
`reference_answer` with `[P1]`, `[P2]` citations in evidence order.
The exporter rejects matching IDs and normalized questions. Keep paraphrases
of benchmark questions out of the training data and review citation support.

```bash
umask 077
APOLLO_CORPUS_DIR="$PWD/data/private/corpus-v6" docker compose run --rm -T \
  -v "$PWD/data/private/sft-train.json:/train.json:ro" \
  -v "$PWD/docs/serbian-support-cases.json:/holdout.json:ro" \
  assistant export-sft --cases /train.json --holdout /holdout.json \
  --corpus /corpus --num-predict 512 > data/private/sft-train.jsonl
```

The export runs offline and contains private source excerpts and reviewed
answers in chat-message format.

## Interpret the comparison

The reports record the corpus fingerprint, embedding digest, case-set hash and system prompt hash. Answer runs also record the chat model and digest. Keep the questions, model and prompt fixed for a corpus or ranker comparison. When testing a prompt change, compare the same fixed-evidence case first and check that the same passages reach both prompts.

- **Evidence absent from the index.** Check for missing documents or unindexed source lines.
- **Evidence indexed but absent from the ranked results.** Inspect retrieval and ranking.
- **Evidence ranked but absent from the prompt.** Check context size and passage selection.
- **Evidence supplied but the answer is wrong.** Review the prompt and model output against the source.

The case file has 35 questions, 23 with reviewed line anchors and seven with reference answers. It covers Serbian Latin and Cyrillic, exact form codes, missing live information, and an out-of-scope control. The cases without anchors include workflows that still need version-matched sources and negative controls that should ask for information or decline. Treat a line overlap as a retrieval signal. Grade each answer for required points, unsupported claims, citation support and handling of missing information. Review ITEM-56's stock-account rule and imbalance separately. Use the current set for development and set aside reviewed questions that the prompt and ranker were never tuned against before reporting a quality score.

## Chart of accounts reference

The supplied TXT identifies the Serbian regulation in Official Gazette RS 89/2020, scoped to the entities named in the regulation. The baseline questions Q01 and Q02 have source anchors in that exact TXT. The regulation gives accounting categories. Your ERP document model, the transaction details and your internal chart are needed for a concrete posting. For live compliance check the applicable version and date with official sources. [Ministry of Finance publication](https://mfin.gov.rs/sr/propisi-1/pravilnik-o-kontnom-okviru-i-sadrzini-racuna-u-kontnom-okviru-za-privredna-drustva-zadruge-i-preduzetnike-sluzbeni-glasnik-rs-br-892020-1)

Put `PRAVILNIK O KONTNOM OKVIRU.txt` from the local private docs into the server app's `data/inputs/` under the same name. Run once to prepare and index the separate reference. Keep the historical journal and the product manual baseline independently available.

```bash
mkdir -p data/private .cache/huggingface
APOLLO_CORPUS_DIR="$PWD/data/private" docker compose run --rm \
  -v "$PWD/data/inputs:/inputs:ro" \
  -v "$PWD/data/private:/prepared" \
  -v "$PWD/.cache/huggingface:/tokenizer-cache" \
  assistant prepare --manual '/inputs/PRAVILNIK O KONTNOM OKVIRU.txt' \
  --output /prepared/chart-accounts-v1 --tokenizer-cache /tokenizer-cache

APOLLO_CORPUS_DIR="$PWD/data/private/chart-accounts-v1" docker compose run --rm \
  assistant index --corpus /corpus --collection apollo-chart-accounts-v1 --kind manual

APOLLO_CORPUS_DIR="$PWD/data/private/chart-accounts-v1" docker compose run --rm -T \
  -v "$PWD/docs/serbian-support-cases.json:/cases.json:ro" \
  assistant evaluate --cases /cases.json --case Q01 --case Q02 --ranker hybrid \
  --corpus /corpus --collection apollo-chart-accounts-v1 --kind manual \
  > data/evaluation/chart-accounts-v1.json
```

## RAGChecker interchange

The existing CLI has all it needs for the baseline. For an answer run, the local report's `ragchecker` field uses the upstream `results` schema for cases with a reference answer and available evidence. Its `retrieved_context` is what Qwen actually received. Extract it from the report with `jq '.ragchecker' report.json` if testing the external tool. RAGChecker is pip-installable in a separate environment. It uses claim extraction and entailment models, and the Quick Start uses English spaCy and an AWS example. Validate the judge on Serbian manual judgments and keep the report local before relying on those scores. [RAGChecker input schema and setup](https://github.com/amazon-science/RAGChecker#-quick-start)

[Chunking paper notes](../knowledge/summary_chunking_reliable_qa.md)

