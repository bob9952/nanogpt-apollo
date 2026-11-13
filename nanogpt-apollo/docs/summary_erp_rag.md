# ERP RAG paper: findings and nanochat experiment boundaries

Prepared: 2026-11-09. Source: arXiv:2511.16700v1, RAG-Driven Data Quality Governance for Enterprise ERP Systems.

## Sources inspected

- Local source archive: [arXiv-2511.16700v1.tar.gz](./arXiv-2511.16700v1.tar.gz); entrypoint Paper.tex.
- Local PDF: [RAG-Driven Data Quality Governance for Enterprise ERP Systems.pdf](<./RAG-Driven Data Quality Governance for Enterprise ERP Systems.pdf>) (present; findings below were checked against the TeX source).
- Public paper: https://arxiv.org/html/2511.16700v1
- Local nanochat checkout: [nanochat](../../nanochat/); CPU entrypoint runs/runcpu.sh, training driver scripts/base_train.py, dataset implementation nanochat/dataset.py, SFT driver scripts/chat_sft.py.

The archive was read in memory. TeX compilation and a TeX installation are unnecessary for this workflow. The archive also contains References.bib, IEEEtran.cls and image4.png; it is publication source, not the authors' runnable ERP application.

## What the paper implements

The authors describe two stages: periodic cleaning/synchronization of employee data from Microsoft SQL Server into PostgreSQL, followed by natural-language SQL generation. Cleaning includes translation normalization, spelling correction, entity merging and integrity checks.

For a question, FAISS retrieves validated question-SQL examples. An existing GPT-4o model receives schema descriptions, business rules, retrieved examples and the user's question. SQL is validated before execution and results are formatted; MongoDB stores session and query logs. The paper describes more than 500 curated examples and normally retrieves five. This is retrieval and in-context adaptation, not nanochat pretraining.

## Evidence limitations

Treat the architecture as a reference and the reported performance as author claims, not independently reproduced results. There are material internal inconsistencies in the supplied v1 source:

- Abstract and evaluation prose: 2,847 production queries; Table IV: 347 queries (Paper.tex lines 50-54, 413 and 467).
- Abstract/model comparison: 92.5% valid SQL and 95.1% schema compliance; Table IV: 84.4% valid SQL and 15.6% schema compliance (lines 54, 397-398 and 468-469).
- The few-shot table gives 91.8% validity, while the ablation narrative uses 92.5% for the retrieval baseline; the distinction between evaluated populations is not adequately explained (lines 272 and 498).
- API cost totals are not reconciled with the stated query count and cost per query (line 486).

These issues do not establish that the architecture is unusable; they prevent adopting the headline numbers as a dependable benchmark.

## Application to the intended Apollo assistant

The intended first task is answering questions about Apollo forms, documents and workflows. Relevant knowledge includes SLC001 metadata, backend package/trigger implementation, bindings, documented business rules and worked document examples. Preserve source paths, checkout revisions, form variants and document identifiers in the index. Answers should cite evidence and identify missing implementation.

The paper focuses on employee-data analytics through SQL. It does not validate explanations of Apollo UI/backend behavior, and it does not prove a tiny model can reason correctly about ERP transactions. Live document status queries would require an additional controlled database integration.

A first retrieval evaluation should use a small set of real questions with reviewed answers and source references. Compare answers without retrieved passages against answers with the same relevant passages. Measure correctness, unsupported claims, source accuracy, latency and resource use. Keep evaluation questions/documents separate from any training split.

## nanochat data paths and experiments

Placing PDFs in docs does not automatically train or index them. The current pretraining data loader expects Parquet files with a text column, with a separate validation file. PDF/TeX extraction, text cleanup, document splitting and metadata preservation are preparation steps, not model training. Avoid putting pages from the same document in both training and validation and then treating memorization as generalization.

SFT consumes conversations through task providers and requires a base checkpoint/tokenizer. RAG requires a separate retrieval/indexing component; no general document retriever was found in the inspected nanochat package, scripts or tasks.

For this VM (10 vCPUs, about 19 GiB usable RAM), begin with a short CPU training smoke test and record peak memory, wall time, per-step throughput, validation loss and source revision. Then choose a training budget from measurements. The only tracked difference in runs/runcpu.sh is its executable-bit change from 100755 to 100644; its text is unchanged. Tiny from-scratch training is an educational baseline; it is not expected to outperform the existing quantized Qwen model on ERP questions without evidence.

## Status

The paper source and relevant nanochat data/training entrypoints were inspected read-only. No paper results, database execution, training, or ERP answer-quality improvement have been reproduced by this summary. Subsequent experiment results must be recorded separately.
