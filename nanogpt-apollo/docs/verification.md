# Phase 1 verification

Phase 1 is complete for the validated private v3 corpus. Use only
`data/private/corpus-v3`; do not reuse the failed v1 validation bundle or the
stopped v2 bundle.

## Corpus evidence

- Fingerprint: `11764c32b018c313ebc3f189b3cf6a3a92ac76c5feac95897e732d9825cc6f80`.
- 956 documents: 2 manuals and 954 source files.
- 27,207 chunks: 388 manual chunks and 26,819 source chunks.
- Maximum observed E5 token count: 384, counting passage prefix, section context,
  and special tokens. All chunk counts were verified.
- 328 exclusions.
- The selected Forms analysis SQL contributes 5,144 chunks across 589 files.
- The guide's original lines 443–459 were checked against the chunks. They retain
  LIN001M001 and LIT002M001, including the stock default and item-specific
  override.

Local SVN metadata was available without network access. Per-file working
revision and status were captured. Of the 26,819 source chunks, 14,084 are from
files at working revision r178 and 12,735 are unversioned.

## Validation evidence

Commands were run with the project virtual environment:

- `.\.venv\Scripts\python.exe -m pytest tests/test_inputs.py tests/test_prepare.py -q` —
  51 passed, 1 skipped.
- `.\.venv\Scripts\python.exe -m pytest tests/test_provenance.py tests/test_prepare.py -q` — 20 passed. This overlaps
  the preceding run; do not add the totals.
- With `APOLLO_REAL_TOKENIZER=1`,
  `.\.venv\Scripts\python.exe -m pytest tests/test_tokenizer_integration.py -q` —
  1 passed.

Both manual hashes remained unchanged during validation.

## Scope and limits

This verifies local corpus preparation, provenance, source selection, manual
spans, and tokenizer counts. Ollama semantics and server performance have not
been tested. No Modal spend, SQL execution, or upstream source/manual changes
occurred. This report records evidence only; it does not stage or commit files.

## Chroma + Compose migration

The current backend is Chroma HTTP with explicit Ollama E5 embeddings. The former
SQLite implementation was locally checked before this migration; those historical
checks are **not Chroma acceptance evidence**. Its storage-specific test file and
the SQLite-dependent answer integration case were retired. Existing preparation,
Ollama, and answer-unit tests remain; no new tests were added or run for this
migration, following the requested server-first scope.

The dependency lock resolves `chromadb-client==1.5.9`. Compose pins
`chromadb/chroma:1.5.9`, persists `/data`, and binds Chroma to loopback. The CLI
profile mounts only the private bundle read only and uses the existing host
Ollama. It does not define an Ollama service.

Local checks are limited to dependency lock/sync, Python compilation/imports,
CLI help, installed public-client API inspection, and `docker compose config`.
No Chroma or Ollama endpoint was contacted locally. Linux image build and server
runtime remain pending: the Windows Docker daemon pipe is absent, and Docker
Desktop was not started. The next acceptance step is the README sequence:
build on the Linux server, start Chroma, run doctor, index 388 manual passages,
and retrieve/answer the single stock-account/item-56 question with citations.
The ten-question comparison remains later work.
