# Apollo RAG — Build Plan

Prepared 20 Nov 2026 for Bob. One developer, CPU-only VM, Serbian Latin answers, no live Oracle access. Status: v1, to be revised after Phase 0.

## 1. Goal, scope and success definition

**Goal.** A private assistant that answers Apollo ERP support questions in Serbian Latin, grounded in Serbian and English manuals plus selected source exports, citing manual and lines.

**Hard requirements (from you).**

- Cite manual and lines for every claim.
- Keep installation-specific guidance conditional ("if your installation uses LIN001M001…").
- Ask for missing transaction details instead of guessing.
- No live Oracle access, no ERP actions.
- Runs on 10 vCPU / 19 GiB / no GPU with Docker Compose, Chroma, multilingual E5 and Qwen 3.5 4B in Ollama.

**Non-goals for now.** Fine-tuning, agents, live database lookups, multi-turn memory, a polished UI.

**Success definition (measurable).**

1. Retrieval: full-coverage@budget (all required evidence present in the prompt) rises from 2/20 to a target agreed after Phase 0.
2. Answers: with correct evidence supplied, the answer cites only evidence that exists, asks for every missing slot, uses conditional wording, and contains no Cyrillic.
3. Latency: p95 end-to-end does not regress against the Phase 0 measurement by more than 20%.

## 2. Evidence base: what I checked and how far to trust it

| Source | What I verified | Trust and use |
| --- | --- | --- |
| intfloat/multilingual-e5-large model card and several guides | E5 needs ` query:  ` and ` passage:  ` prefixes, also for non-English text; inputs truncate at 512 tokens. One guide measured large drops when prefixes are wrong. | High for the rule (it is the model card). Audit your code for it first. |
| Qwen3.5 model card | Runs in thinking mode by default; the card recommends large contexts to preserve thinking. | High. For CPU RAG, switch thinking off and measure. |
| Ollama docs via third-party docs | Ollama can fall back to a small default context (4096 mentioned) unless `num_ctx` is set explicitly. | Medium. Verify on your install with a long prompt. |
| Hybrid BM25 + dense + RRF write-ups (several blogs, LangChain/RAGLight docs) | The pattern is standard; RRF merges by rank, constant k=60 is typical. Claimed gains (+15–30%) are blog numbers, not benchmarks. | Pattern: high. Gains: ignore. Measure on your own set. |
| Evaluation write-ups (Strata-RAG wiki, dev.to posts, GuardRAG issue #9) | Measure retrieval and generation separately; gold set mapped to document+section so it survives re-chunking; include out-of-corpus questions; report per language. | Hobby-grade, but the method is sound and consistent across sources. |
| dev.to "measured the RAG technique menu on 46,000 chunks" | Gold spans stored as character offsets make the test set independent of chunking; ingest-time cost is paid once, query-time cost on every request. | Single author, one corpus. Use the ideas, not the conclusions. |
| ProbabilityRAG (Hugging Face space) | A reranker looked bad because the candidate pool was too small; most "model errors" were retrieval errors. | Anecdote that matches the general principle. |
| Serbian NLP survey (arXiv 2304.05468) and `serb_stem` on PyPI | Serbian stemmers are rule-based with modest reported accuracy (79–90% for the older ones); n-gram normalisation is an accepted alternative. `serb_stem` is a young single-author package. | Medium. Do not depend on a stemmer without A/B testing it. |
| Network-Intelligence-Chat (cloned and read) | See my earlier answer: typed query spec, exact lookup before semantic search, separate retrieval and answer evals. | High (read the code). Heavy LLM loops there are not for your hardware. |
| nanochat and modded-nanogpt (README and wiki read) | Both are from-scratch pretraining harnesses targeting 8×H100 nodes (speedruns to GPT-2 quality). | Not applicable to Apollo. Educational only (Section 12). |
| RAG-for-ERP paper (arXiv 2511.16700) | Useful ideas (business-logic encoding, corpus of validated pairs, failure taxonomy) but internal numeric inconsistencies; it is NL-to-SQL, not document RAG. | Low as evidence. |

**Not found.** I could not retrieve the Reddit threads themselves. Reddit pages were not returned by search and a direct fetch was refused, so everything above comes from GitHub, PyPI, dev.to, Hugging Face and arXiv. One r/Rag post ("Fully offline RAG over 27 technical books… with gold-set numbers") is referenced by a GitHub issue; if you want it used, paste its text and I will fold it in.

## 3. Fixed decisions and settings to verify

1. **E5 prefixes.** Index chunks as `passage: <text>`, queries as `query: <text>`. Store the prefix logic in one function used by ingestion and retrieval. Add a unit test that fails if a chunk is embedded without it.
2. **512-token ceiling.** Keep chunks to roughly 350–450 tokens of body text plus a short header, so nothing is silently truncated. Check with the E5 tokenizer, not by word count.
3. **Embedder loads once.** Load at process start, not per query.
4. **Qwen thinking off.** For this task, disable the thinking mode and compare answer quality on the oracle-context set before accepting it.
5. **Explicit context.** Set `num_ctx` explicitly in the Ollama request and confirm with a long prompt that nothing is truncated. Derive the evidence budget from it: `budget = num_ctx − system prompt − question − answer reserve`.
6. **Low temperature** for answers; record every generation parameter in the log.
7. **Reproducibility.** Pin model versions, embedder revision, chunker version and index version; write them into every evaluation report.

## 4. Architecture

```
question
  → query.py        rules first: extract codes, entities, symptoms; list missing slots
  → router.py       per symptom: exact lookup | keyword | dense
  → retrieval/      exact index + BM25 (FTS5) + dense (Chroma) → RRF per symptom
  → evidence.py     per-symptom quota, token budget, provenance
  → answers.py      prompt contract → Qwen → citation validation
  → log (jsonl)     query spec, evidence ids, answer, latency
```

```
apollo_rag/
  schema.py            dataclasses: Chunk, Symptom, QuerySpec, Evidence, EvidencePack
  text.py              normalisation: script, diacritics, case, glossary expansion
  query.py             symptom/entity/slot extraction (rules, optional LLM fallback)
  ingestion/
    parse.py           manuals/exports → sections with line ranges
    chunk.py           structure-aware chunking, headers, provenance
    index.py           builds Chroma, SQLite FTS5, form-code index
  retrieval/
    exact.py           form codes and known identifiers
    sparse.py          FTS5 keyword search
    dense.py           Chroma + E5 (prefix handled here only)
    fuse.py            RRF, per-symptom quota
    router.py
  answers.py           prompt, generation, citation validation
  cli.py
  evaluation/
    gold/              YAML cases
    retrieval_eval.py  no LLM
    answer_eval.py     deterministic checks + human grading sheet
docs/                  runbooks, glossary, benchmark notes
scripts/               server commands
```

Split `retrieval.py` into `retrieval/` only after Phase 0 exists, so the benchmark can catch regressions.

## 5. Data contracts

**Chunk** (stored in Chroma metadata and in SQLite):

- `chunk_id`, `doc_id`, `source_type` (`manual` | `export` | `ticket`), `language`, `section_path`, `line_start`, `line_end`, `char_start`, `char_end`, `form_codes[]`, `installation` (null for manuals), `version`.
- `text_raw` (what you cite) and `text_embed` (header + text, what you embed).

**QuerySpec**:

- `symptoms[]`: each with `id`, `kind` (e.g. `posting_imbalance`, `account_determination`), `terms[]`, `entities` (item=56, …).
- `known_slots{}` and `missing_slots[]`.
- `language`, `script`.

**EvidencePack**: per symptom, ordered `Evidence(chunk_id, citation, source_type, text, scores)`; plus total tokens and which gold-relevant fields are covered.

**Gold case** (YAML, survives re-chunking):

```yaml
id: T-0007
question: "..."
symptoms:
  - id: s1
    required_evidence:
      - {doc: manual_x, section: "5.3", lines: [120, 148]}
  - id: s2
    required_evidence:
      - {doc: export_y, form_code: LIN001M001}
missing_slots_expected: [warehouse, item_type, debit_credit_lines]
language: sr-Latn
tags: [two-symptom, installation-specific]
```

A retrieved chunk counts as a hit if its line range overlaps a required range. That keeps gold data valid when chunking changes.

## 6. Phases

Effort figures are rough part-time estimates for one developer; revise after Phase 0.

### Phase 0 — Instrument before changing anything (2–3 days)

- Convert the 20 cases to the gold format above.
- Write `retrieval_eval.py`: full-coverage@budget, per-symptom coverage, recall@k per evidence item, MRR, evidence tokens, per-language and per-tag slices. No LLM involved.
- Classify every miss: not retrieved in top 50; retrieved but cut by budget; chunked badly (evidence split across chunks); not indexed as expected.
- Measure baseline latency: retrieval p50/p95, generation tokens/s, end-to-end.
- **Done when:** the script reproduces 2/20 exactly, deterministically, in under two minutes, and the miss classes are tabulated.

### Phase 1 — Ingestion hygiene and prefix audit (3–4 days)

- Fix E5 prefixes if absent; load the embedder once.
- Parse manuals into sections with stable line ranges (the same numbering you cite to users).
- Structure-aware chunking: split on headings and procedure steps, never mid-step; chunk body ≤ \~400 tokens; prepend a deterministic header ("Manual › Section › Form code") to the embedded text only.
- Link Serbian and English versions of the same section via `section_path` or an explicit map.
- Build the form-code index: every code like `LIN001M001` → chunks that define or mention it, with its key fields.
- Build the glossary (konto/account, magacin/warehouse, vrsta artikla/item type, …) as a plain file in `docs/` that both queries and chunks can use.
- **Done when:** every chunk resolves to manual + lines; no embedded text exceeds 512 tokens; the re-run benchmark is no worse than baseline.

### Phase 2 — Symptom-aware retrieval (4–6 days)

- `query.py`: regexes for codes, "item N", "stock account", "out of balance", plus glossary terms; split a ticket into symptoms; compute missing slots.
- `sparse.py`: SQLite FTS5 over normalised text (Section 9). No extra service.
- `fuse.py`: run dense and sparse per symptom with a candidate pool of 30–50, merge with RRF, then apply a per-symptom quota inside the token budget.
- `exact.py`: direct hits for known codes and identifiers always enter the pack.
- Optional, only if recall@50 is high but recall@budget is low: a multilingual cross-encoder reranker on the top 20–30 (the ProbabilityRAG lesson: give it a pool big enough to contain the answer). Measure its latency on your CPU before accepting it.
- **Done when:** full-coverage rises materially from 2/20 (target set after Phase 0; my placeholder is ≥10/20), no tag slice gets worse, and the ticket case has both symptoms covered.

### Phase 3 — Answer contract (3–4 days)

- Evidence pack formatted per symptom with citation labels `[manual:L120–148]`.
- Missing slots computed in code and passed to the prompt as a list the answer must ask about.
- Rules in the prompt: answer only from evidence; mark anything sourced from an export as installation-specific and conditional; if a symptom has no evidence, say so and ask for what is needed; reply in Serbian Latin.
- `answers.py` validates after generation: every cited range exists in the pack; no form code appears that is absent from the pack; no Cyrillic. On failure, one repair attempt, then return a safe fallback.
- **Done when:** on the oracle-context set, 100% of citations resolve, zero hallucinated codes, zero Cyrillic, every expected slot question appears.

### Phase 4 — Reviewed ticket corpus (ongoing)

- Store human-reviewed ticket → resolution pairs as a separate index (`source_type=ticket`, `installation` tag, `review_status`).
- Only reviewed pairs are retrievable; resolutions are shown as "previously resolved in installation X", never as manual truth.
- Redact customer and personal data before indexing.
- **Done when:** adding a pair measurably changes a held-out case, and nothing unreviewed can appear in an answer.

### Phase 5 — Hardening (as needed)

- Query-time cost review (anything that costs a Qwen call per request must earn its place).
- Caching of embeddings for repeated questions.
- Reindex command and index versioning; a regression run on every change.

### Phase 6 — Fine-tuning gate (only if Section 12's conditions hold)

## 7. The reference ticket, end to end

Ticket: a document is out of balance; Apollo could not find a stock account for item 56.

1. **Query understanding** produces two symptoms: `posting_imbalance` and `account_determination` (entity: item 56). Missing slots: warehouse, item type, document type, debit and credit lines.
2. **Routing.** The second symptom goes through the form-code index and keyword search for the stock-account concept and finds `LIN001M001`, keyed by warehouse and item type. The first goes through dense plus keyword retrieval over posting and balancing sections.
3. **Fusion and quota.** Each symptom gets its own evidence slots, so a strong match for one cannot crowd out the other. My working hypothesis for the current 2/20 is exactly that crowding in a single top-k pool; Phase 0's miss classes will confirm or refute it.
4. **Evidence pack.** Manual passages for both symptoms with line citations, and any export row marked installation-specific.
5. **Answer (Serbian Latin).** Explain both symptoms separately; say that if the installation uses `LIN001M001`, the stock account is found by warehouse and item type, so check that row for the warehouse and item type of item 56; list the missing details (warehouse, item type, the actual debit and credit lines) and explain why the imbalance cannot be diagnosed without them.
6. **Checks.** Citations resolve; `LIN001M001` is present in the pack; all three slot questions are asked; wording is conditional.

## 8. Evaluation design

**Gold set.** Grow from 20 to roughly 40–60 cases. With 20 cases each case is five percentage points, so small improvements are noise. Include: two-symptom tickets, single-symptom tickets, questions with the answer only in an export, out-of-corpus questions (the assistant must say it lacks evidence), questions typed without diacritics, Cyrillic input, and English questions against Serbian manuals. Label how each case was created (written by you from real tickets vs generated and reviewed).

**Retrieval metrics (no LLM).**

- full-coverage@budget (the headline: all required evidence in the prompt),
- per-symptom coverage, recall@k for k in 5, 10, 30, 50, MRR,
- evidence tokens used,
- slices by language, tag, source type.

**Answer evaluation, in two modes.**

- *Oracle context*: feed gold evidence directly to isolate the generator.
- *End to end*: real retrieval. The gap between them is your retrieval loss.

**Deterministic answer checks.** Citations resolve; cited text supports the claim (spot-check by hand); no unknown form codes; no Cyrillic; every expected slot question present; conditional phrase present for export-sourced statements; abstention when evidence is absent.

**Human grading.** Grade 20 answers by hand on a fixed rubric (correct, supported, complete, asks the right questions). Do not use Qwen 4B as a judge; it is not a trustworthy instrument.

**Regression gate.** `scripts/eval.sh` runs retrieval eval on every change and fails if full-coverage or any slice drops beyond a threshold you set. Retrieval eval must stay cheap enough to run each time.

## 9. Serbian text handling

- Normalise at both index and query time: unify to Latin script, lowercase, fold diacritics (č ć š ž đ), handle "dj" versus "đ", collapse spacing and punctuation. Users often type without diacritics.
- Keep `text_raw` untouched for citation and display; use normalised text only for search.
- Keyword search: start with FTS5 on normalised text with prefix queries; A/B against a trigram tokenizer and against a stemmer (`serb_stem` or a simple conservative prefix stem). Adopt a stemmer only if it wins on your gold set.
- Glossary expansion at query time for ERP terms in both languages; keep it a reviewed text file, not model-generated.
- Quote identifiers (form codes, account numbers) exactly; never normalise them.

## 10. Generation and latency budget

- Prompt order: system rules, symptoms and missing slots, evidence per symptom, then the question. Keep the pack short; more chunks added confusion with a small local model in at least one write-up I read, so measure the budget instead of maximising it.
- Measure tokens/s on your VM and set the maximum answer length from it. Decide the latency target after Phase 0, not before.
- Everything that can happen at ingest time (headers, glossary links, form-code index) should; every query-time LLM call must justify its cost.

## 11. Operations, privacy and logging

- Log per request as JSONL: query spec, evidence ids and scores, prompt tokens, answer, validation results, latencies, versions.
- Keep logs and tickets inside your infrastructure; redact before any indexing.
- Docker Compose: one service for the API, Chroma, Ollama; pinned images; index built by a script, not at start-up.
- Backups: the index is rebuildable, the gold set and reviewed ticket corpus are not; version them in SVN or git.

## 12. Fine-tuning gate and the nanochat question

**nanochat and modded-nanogpt are not for this project.** They are minimal harnesses for pretraining small GPT-style models from scratch on 8×H100 nodes, tuned for training speed. They teach how models are trained; they will not improve Apollo retrieval or answers. Read them for understanding if you like, not as a build input.

**If fine-tuning ever becomes justified**, the practical route is parameter-efficient (LoRA-style) supervised tuning of Qwen on Modal, then conversion back to something Ollama can serve on CPU. Preconditions:

1. Oracle-context answers still fail on a consistent behaviour (Serbian grammar, skipping slot questions, losing conditional wording).
2. Prompt changes, few-shot examples and the answer contract have plateaued.
3. You have a few hundred reviewed gold answers and a held-out set never used for training.
4. The tuned model beats the baseline on the held-out set without losing citation validity, and still fits your CPU latency.
5. Policy check: may ticket data leave your infrastructure for Modal at all?

If retrieval is the bottleneck, tune the embedder or reranker with query–passage pairs first; it is cheaper and addresses the real fault.

## 13. Risks

- Gold set too small or labelled loosely → mitigate by growing it and labelling by section and lines.
- Chunking splits a procedure across chunks → structure-aware chunking and overlap on steps; Phase 0 classifies these misses.
- Export rows treated as general truth → source-type tagging and conditional wording enforced by validation.
- Quality gains from reranking or stemming that vanish on your data → adopt only with an A/B on the gold set.
- Ollama context truncation → explicit `num_ctx` and a long-prompt test.
- Single-developer bus factor → keep runbooks in `docs/`.

## 14. What I need from you to sharpen this

1. The 20 benchmark cases (question, required evidence) and the current `retrieval.py`.
2. Manual formats (PDF, HTML, text?) and how line numbers are produced today.
3. Export formats and where `LIN001M001` actually lives (manual, export, both).
4. Current chunk size, overlap, top-k, and the prompt token budget; the Ollama request settings.
5. Corpus size (manuals, pages, languages, Cyrillic present?).
6. Volume of resolved tickets available for Phase 4.
7. Your latency expectation per answer.

## 15. Reading list

- intfloat/multilingual-e5-large model card (prefix rules, 512-token limit).
- Qwen3.5 model card (thinking mode, recommended sampling).
- Network-Intelligence-Chat repository (typed query spec, evaluation split).
- Strata-RAG wiki, "Evaluation" (retrieval vs generation).
- GuardRAG issue #9 (gold set by document and section, per-language recall).
- dev.to, "I measured the RAG technique menu on 46,000 chunks" (offset-based gold spans).
- ProbabilityRAG README (candidate pool size for rerankers).
- arXiv 2304.05468, survey of Serbian NLP resources.
- arXiv 2401.14887, "The Power of Noise" (effect of irrelevant passages on RAG).
- arXiv 2403.09040, "RAGGED" (how much context readers can use).
- karpathy/nanochat and KellerJordan/modded-nanogpt (pretraining background only).
