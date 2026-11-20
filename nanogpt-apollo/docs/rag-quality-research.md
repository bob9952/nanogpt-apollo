# Apollo ERP RAG research

The 4,275 chunks in the deployed v5 manuals are enough to measure a useful support assistant. The failed item 56 query shows a ranking problem and a gap in the evidence passed to Qwen. Build the expanded v6 manuals, compare the same questions across both collections, and use the results to decide what to change. Keep the existing Qwen 4B and E5 models fixed for this comparison.

## What the logs and source show

- For the direct stock-account question, v5 found the G4 guide's default rules for warehouse and item type in `LIN001M001` at lines 443–459 and the item-specific setup in `LIT002M001` at lines 564–591. The Qwen run that received only the first passage still named both forms because that passage mentions the item override. The second passage describes the form in more detail.
- For the complete item 56 support request, v5 sent exchange-rate setup and the stock-card display to Qwen. Neither passage supported the suggested cause. In Apollo's current lexical scoring, each distinct matching term adds a whole point while vector rank contributes at most 1/61. Generic posting language can outrank a passage with the right stock-account rule. Exact form codes get an even larger boost. `cosine=n/a` means a result had no cosine recorded in the vector candidate window. A cosine near 0.88 can occur for unrelated questions because E5 scores naturally cluster high. [Apollo retrieval code](../apollo_rag/retrieval.py#L52) · [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small#faq)
- The context-selection code compares the number of UTF-8 bytes with the 4,096-token generation budget. This conservative estimate can skip entire relevant passages, especially when long JSON-formatted evidence or multibyte characters are present. The earlier log explicitly reported using one of two passages. Track the retrieved passages and the passages actually delivered to Qwen separately. [Apollo prompt construction](../apollo_rag/answers.py#L28) · [Ollama settings](../apollo_rag/ollama.py#L13)
- The archived local v3 bundle demonstrates why totals mislead. Of 27,207 chunks, 388 are manual passages and 26,819 are source passages. The median manual passage contained 105 E5 tokens including its embedding prefix and section, and 46 manual passages contained only their heading. These are measured v3 figures. The 4,275 figure for v5 comes from the Ubuntu index log, and the v6 chunk count and ranking still need an Ubuntu run.

## Which documents earn a place

| Source | What it can answer | How to use it |
| --- | --- | --- |
| Apollo G4 guide and Serbian Apollo manuals | Default accounts, form fields, user steps | Core support collection. Keep the form ID, product generation, customer or installation where relevant, and original page or line range. |
| Lovcen Auto v1.0 manual | What the documented financial rule does when the posting account is blank | V6 adds this source. Section 3.4.5, lines 1733–1738, describes the stock-state lookup by item, organizational unit, validity date and partner, followed by the default lookup via `LIN001M001` when the state is missing. Establish whether that rule applies to the reported document before diagnosing it. |
| Procurement manual | Balance checks in a documented dependent-cost workflow | V6 adds this source. Lines 1270–1273 describe a rounding correction in that specific workflow. Require evidence about the document type before using that example. |
| May 2022 Privredni savetnik e-invoicing extract | Historical SEF launch and XML upload issues | Useful dated reference. The prepared ten-page text keeps the PDF page boundaries and fixes multi-column reading order. Check dates, exceptions, and table cells against the original before using them. |
| February 2015 Privredni savetnik OCR | Historical inventory accounting examples, including write-downs | Keep in the separate accounting collection. Verify figures and account labels against the scanned page, especially when the model cites a table. |
| Longer May 2022 Poslovni savetnik extract | Additional historical accounting and e-invoicing topics | Found locally and absent from the v6 archive. It has about one million text characters. Select sections tied to benchmark questions, check tables against the PDF, and check for overlapping articles before indexing. |

Dates, installation names, document model IDs, and applicability are worth more than sheer volume. Prioritize reviewed support cases with the exact error, what the user was trying to do, and the verified resolution. Include official and date-matched sources for questions about current SEF rules. An old journal and a product manual answer different parts of a support question. Preserve original references when extracting tables or reorganizing scanned pages.

The ERP data-governance preprint is useful as an engineering example. It describes normalization, curated examples, and separate SQL validity, schema compliance, and semantic accuracy measurements. Its task was validated multilingual SQL over HR data. Apollo's current task is cited manual-based troubleshooting, so carry over the evaluation discipline and use Apollo's own support checks. [ERP data-governance preprint](https://arxiv.org/abs/2511.16700)

## What to take from Aleksandar and Karpathy

**Aleksandar's team.** The checked-out Network Intelligence Chat at `af227b0` models a query first, chooses structured filters, semantic retrieval or a hybrid union, then synthesizes an answer. The retrieval graph evaluates and can rerank results. Its hybrid code deduplicates profiles by LinkedIn URL. The repo includes 18 parsing cases and 30 retrieval cases with known expected IDs. The evaluation measures retrieval precision and recall, and a separate judge covers groundedness and completeness. I inspected the implementation and case files. I did not run or score their system. For Apollo, map the same ideas to exact form lookups, localized symptom searches and expected source spans. Keep the CPU path small by starting with deterministic routing. [Project repository](https://github.com/acailic02/Network-Intelligence-Chat) · [Local retrieval evaluation](../../Network-Intelligence-Chat/evaluation/retrieval_eval.py) · [Local synthesis evaluation](../../Network-Intelligence-Chat/evaluation/llm_judge.py)

**Karpathy.** Nanochat trains a model from scratch and evaluates checkpoints with measures such as validation bits per byte and DCLM CORE. Its CPU script is a small educational training run. Copy the fixed baseline and one-change-at-a-time experiment loop. Measure Apollo on real support questions using its existing pretrained Qwen. A training loss or a generic CORE score gives no evidence of whether a posting answer cites the right manual section. [Nanochat README](https://github.com/karpathy/nanochat/blob/master/README.md) · [CPU example](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh)

## First experiment on Ubuntu

1. Build and index the ERP part of the private v6 archive. Compare v5 and v6 with the same installed app, Ollama tags and questions. Keep v5 available as the baseline.
2. Create 20–30 reviewed Serbian questions from support tickets. Cover exact form IDs, natural phrasing with and without diacritics, Cyrillic, multi-symptom tickets, similar G4/G5 or customer variants, and unanswerable live-data questions. Keep variants of one incident together when splitting development and holdout cases.
3. Label expected evidence by document and page or line range. Measure whether at least one relevant passage appears in the first five results, how much required evidence appears in the first ten, and whether all required evidence reaches the Qwen prompt. Review supported claims, Serbian Latin, citation accuracy and appropriate requests for missing data. Log search and warm-answer latency separately. Known spans survive rechunking and duplicate chunks earn credit only once. [RAGChecker](https://arxiv.org/abs/2408.08067) · [ALCE citation evaluation](https://aclanthology.org/2023.emnlp-main.398/)
4. Test lexical ranking as the first code change. Compare E5 alone, normalized BM25 with exact form and error phrase handling, and reciprocal rank fusion of the two lists. Preserve a strong exact-match route for identifiers such as `FDC001M001`. Try two focused searches for multi-symptom questions, merge and deduplicate evidence, and keep the most useful two to four passages in the answer. These are starting settings to measure on the VM. Add a reranker only if relevant passages regularly appear in the candidate list and still miss the final context. [BEIR](https://arxiv.org/abs/2104.08663) · [Reciprocal rank fusion](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf)
5. Test Qwen once with reviewed evidence supplied directly. A good fixed-evidence answer points back to retrieval or context selection. A poor fixed-evidence answer gives a reason to adjust the prompt, evidence format or generator. Consider model changes only after that control.

Keep the current 384 E5-token ceiling first. Check whether retrieval loses short bullets or procedures split across adjacent chunks. If it does, A/B test grouping coherent paragraphs or numbered steps within one heading, retaining form IDs and source spans. Start with roughly 200–350 E5 tokens where the content naturally fits. The prefix, heading, and special tokens count toward the cap. Published experiments found different strong chunking methods under different setups, so treat passage shape and context size as measurements. [Sentence chunking study](https://arxiv.org/abs/2601.14123) · [Paragraph grouping study](https://arxiv.org/abs/2603.06976) · [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small)

### Cases to start with

| Serbian support question | Expected evidence or behavior |
| --- | --- |
| Gde se podešava podrazumevano konto zaliha za magacin i tip artikla, a gde izuzetak za artikal? | G4 guide 443–459 and 564–591. Distinguish `LIN001M001` defaults from `LIT002M001` item setup. |
| Zašto ne mogu da knjižim dokument? Nije u ravnoteži i nije nađeno konto zaliha za artikal 56. | Find the Lovcen lookup rule and G4 account setup. Treat the balance symptom separately. Ask for the document type, organizational unit and exact errors or financial lines before identifying a live cause. |
| Šta se dešava kad u FDC001M001 ostavim konto prazno? | Lovcen Auto v1.0 section 3.4.5, lines 1733–1738. Explain the documented rule and preserve the form ID. |
| Koji je stvarni broj konta zaliha za artikal 56 u mom magacinu? | Request the organizational unit and actual stock or setup record. The manuals cannot confirm the configured number. |
| Koliko pomorandži stane u Citroën C3? | Decline as outside the Apollo corpus, even when the embedding search returns passages. |

Once v6 has been indexed, this compares the original incident on both collections without invoking Qwen:

```bash
cd ~/ai/nanogpt-apollo/nanogpt-apollo
question='Zašto ne mogu da knjižim dokument? Dokument nije u ravnoteži i sistem nije našao konto zaliha za artikal 56. Šta da proverim?'
for version in v5 v6; do
  printf '\n=== %s ===\n' "$version"
  APOLLO_CORPUS_DIR="$PWD/data/private/corpus-$version" \
    docker compose run --rm assistant search "$question" \
      --corpus /corpus --collection "apollo-manuals-$version" --kind manual --top-k 10
done
```

## The Transformers warning

Apollo loads the E5 fast tokenizer inside the Python container and explicitly disables the Torch, TensorFlow and Flax backends. Transformers 4.57.1 reports that those model runtimes are unavailable. The tokenizer still counts and splits passages. Ollama runs the embedding model and Qwen in its own process, and Chroma stores the vectors. A final `Prepared … chunks` line confirms preparation. A separate successful `Indexed … chunks` line confirms indexing. The `521 > 512` tokenizer message is a different warning. Preparation measures candidate text with truncation disabled, including candidates too large to keep, then bounds the emitted passages. Ollama also rejects oversized embedding inputs instead of silently truncating, and the indexer has a split-and-retry path. Investigate if preparation or indexing actually terminates with an error. [Apollo tokenizer](../apollo_rag/tokenizer.py) · [Passage splitter](../apollo_rag/chunks.py#L82) · [Transformers warning source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/__init__.py#L3853-L3861) · [Ollama embedding API](https://docs.ollama.com/api/embed)

This report uses the local repo, archived corpus evidence and supplied Ubuntu logs. The v6 comparison and proposed ranking experiment remain unmeasured.
