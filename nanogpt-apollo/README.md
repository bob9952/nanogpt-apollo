# Apollo ERP RAG

Private Apollo manual/source retrieval using **Chroma + existing Ollama**.
Chroma stores explicit E5 vectors; Ollama provides embeddings and streamed Qwen
answers. The app validates the portable corpus and maps results back to its full
passages, chunk IDs, paths, and line/page ranges. No model training is involved.

The workflow is `prepare` (raw inputs to a corpus), `index` (corpus to Chroma),
`search` (original source passages), then `ask` (a cited Serbian Latin answer).
The accepted v3 corpus is already prepared, so skip `prepare` for the first run.
`search` does not generate or translate an answer: English manual text remains
English. `ask` instructs Qwen to answer in Serbian Latin while preserving form
codes; this instruction still needs to be checked against its actual output.

## First run on the Linux server

Run every command in this section on the Ubuntu server through PuTTY. Python 3.12
and the pinned dependencies are installed inside the Docker image; no host Python
or `uv` installation is required.

The Git repository is `bob9952/nanogpt-apollo`. Its root contains the application
subfolder **`nanogpt-apollo/`**. From a parent directory:

```bash
git clone --branch feat/apollo-chroma-compose https://github.com/bob9952/nanogpt-apollo.git apollo-work
cd apollo-work/nanogpt-apollo
```

For an existing clone, run from its repository root:

```bash
git fetch origin
git checkout feat/apollo-chroma-compose
git pull --ff-only
cd nanogpt-apollo
```

Use WinSCP to transfer the **complete** private `data/private/corpus-v3` bundle
separately into this application's `data/private/corpus-v3/`. Copy its manifest,
chunks, exclusions, sample passages, and other bundle files together. These files
stay outside Git and the Docker build context. The validated v3 bundle has 956
documents and 27,207 passages, including **388 manual passages**. Original Windows
paths in the manifest are provenance; the server never opens those original files.

Compose requires Linux Docker Engine and the existing host Ollama endpoint.
It starts only Chroma; the assistant is a CLI profile. It does not create or
change an Ollama container. The corpus bind is read only and a missing source
folder causes an error instead of silently creating an empty directory.

```bash
cp .env.example .env
# Edit APOLLO_CORPUS_DIR only if the complete bundle lives elsewhere.
docker compose build assistant
docker compose up -d chroma
docker compose run --rm assistant doctor
```

Doctor checks Chroma heartbeat and these exact installed Ollama tags:

- `qwen3.5:4b-q4_K_M`
- `qllama/multilingual-e5-small:q8_0`

If Chroma is still starting, inspect `docker compose logs chroma`, then rerun
doctor. If E5 is missing and the existing Ollama container is named `ollama`,
pull it explicitly and rerun doctor:

```bash
docker exec ollama ollama pull qllama/multilingual-e5-small:q8_0
```

The app never downloads a model. Use the existing Ollama installation's command
if its container has a different name. Keep the existing Qwen tag above installed.

Start with the 388 manual passages and **one question**. Run `index` only if the
v3 collection has not already been indexed:

```bash
docker compose run --rm assistant index --corpus /corpus --collection apollo-manuals --kind manual
docker compose run --rm assistant search "zasto nece da mi knjizi dokument, nije mi u ravnotezi, nije mi nasao konto zaliha za artikal 56?" --corpus /corpus --collection apollo-manuals --kind manual
docker compose run --rm assistant ask "zasto nece da mi knjizi dokument, nije mi u ravnotezi, nije mi nasao konto zaliha za artikal 56?" --corpus /corpus --collection apollo-manuals --kind manual
```

The first server search returned only manual section headings (guide lines 1694,
2080, and 2646), so retrieval quality has not been established. The application
now excludes manual passages whose complete text is only their section heading,
in both semantic and literal rankings. It expands the candidate window when
needed, using the same query vector. This filter works with the existing v3 bundle
and collection: rebuild the assistant image and rerun the exact `search` and `ask`
commands above; no preparation or reindexing is needed for this code update.

Check the retrieved passages and answer citations against the manual. The stock
account evidence is in guide lines 443-459: LIN001M001 provides warehouse/item-type
defaults and LIT002M001 an item override. Balance status is separate evidence in
lines 2252-2268. Filtering headings does not guarantee relevant E5 rankings;
retrieval of these passages and answer quality still need server verification. The answer
must distinguish a documented rule from the actual item-56 setting on a specific
installation. Report any missing evidence instead of assuming the setting. Review
retrieval time, first-answer time, and generation throughput printed by `ask`.
A ten-question comparison can follow after this first server question works;
there is no automated evaluation command in this iteration.

## Storage and retrieval

Chroma server and thin HTTP client are pinned to `1.5.9`. Chroma data persists in
the Compose named volume `chroma-data` mounted at `/data`. Normal container
recreation retains it; `docker compose down -v` deletes it. Chroma publishes only
`127.0.0.1:${CHROMA_PORT:-8000}`. The CLI uses host networking, so it connects to
`127.0.0.1` for Chroma and the existing Ollama, not Compose service DNS.

`.env` configures `OLLAMA_URL`, `CHROMA_HOST`, `CHROMA_PORT`, and
`APOLLO_CORPUS_DIR`. CLI overrides are `--ollama-url`, `--chroma-host`, and
`--chroma-port`. Compose disables proxy routing for these explicit endpoints.
Passage and question text goes only to the configured services; keep them private.

Indexing sends the stored `passage: ` embedding inputs unchanged to Ollama and
upserts explicit vectors, full documents, IDs, and simple metadata into a cosine
collection. Chroma uses `embedding_function=None`; it does not load a default
embedding model. The thin client avoids the embedded server/ONNX dependencies.
Client telemetry is disabled.

Matching existing IDs are reused when indexing resumes. Collection metadata binds
the corpus fingerprint, E5 digest, dimension, algorithm, scope, and completion
state. Interrupted collections cannot be searched until the same index command
completes. A changed corpus, model digest, or scope requires a **new collection
name**; the app never deletes or resets an existing collection. Run one indexer
per collection at a time. Batches already stored survive an interrupted run;
unfinished embedding requests can be repeated.

Search validates the bundle hashes/identities and collection metadata before
querying with an explicit `query: ` embedding. It combines Chroma semantic ranks
with a small literal lookup for Apollo identifiers containing digits or underscores
(such as `LIN001M001` and `WX_LIB2`). The literal lookup scans the already loaded
corpus; it adds no second storage backend. A literal-only hit prints cosine `n/a`.
The default returns three whole passages. `--top-k` changes that count;
`--kind manual|source|both` filters retrieval. A manual-only collection requires
`--kind manual`. To include source later, index a new collection with `--kind both`.
There is no SQLite fallback or application-managed vector matrix.

Ollama embedding truncation is disabled. If an input is oversized, bounded batch
and passage splitting pools child vectors while retaining every character in the
original full passage and citation. Oversized questions fail with a request to
shorten them. Ask keeps the existing conservative context budget and streams the
existing Qwen request; omitted passages, missing/unknown citations, partial streams,
and generation cutoffs are reported. Citations identify evidence, but do not prove
that a generated claim is supported. Read the referenced passages.

## Server-side preparation (optional for new inputs)

Skip this section for the initial v3 run. Preparation requires the **raw UTF-8 TXT
manuals**; the v3 bundle's chunks are not those input files. After initial v3
setup, use WinSCP to copy both raw manuals into the ignored `data/inputs/`
directory, preserving these exact names and capitalization:

- `Apollo G4 ERP Users Guide v2.txt`
- `specification of app english.txt`

From the application folder on the Ubuntu server, confirm that both files are
present before running `prepare`. If they are missing, transfer them first:

```bash
mkdir -p data/inputs data/private .cache/huggingface
ls -l data/inputs
```

Then prepare a new bundle. In the input mount below, the colon immediately follows
`inputs`; do not add a slash or a Windows backslash before it:

```bash
docker compose run --rm \
  -v "$PWD/data/inputs:/inputs:ro" \
  -v "$PWD/data/private:/prepared" \
  -v "$PWD/.cache/huggingface:/tokenizer-cache" \
  assistant prepare \
  --manual '/inputs/Apollo G4 ERP Users Guide v2.txt' \
  --manual '/inputs/specification of app english.txt' \
  --output /prepared/corpus-v4 \
  --tokenizer-cache /tokenizer-cache
```

The existing read-only v3 corpus bind remains required by Compose, so this optional
preparation step is for use after the initial v3 setup. Choose a fresh output name
that does not already exist; do not delete an existing corpus. For a supplied
source export, add `--source-root /inputs/source-export`; it must be an exported
directory because the server cannot access original Windows or UNC paths. If the
export has no SVN metadata, that provenance is reported as unavailable. The first
online run downloads public tokenizer files into the cache; use `--offline` only
after those files are cached. This does not train a model. Research files in `docs/`
remain background material and are not default corpus inputs.

New preparation keeps each manual heading with its following body across blank
lines and page breaks, while retaining original source spans. This changes the
prepared corpus. For the new bundle, set
`APOLLO_CORPUS_DIR=./data/private/corpus-v4` in `.env` and index it into a new
collection:

```bash
docker compose run --rm assistant index --corpus /corpus --collection apollo-manuals-v4 --kind manual
```

No additional Compose services or configuration are needed. Read the prepared
bundle's text for a quality check; no preparation test command is required.

