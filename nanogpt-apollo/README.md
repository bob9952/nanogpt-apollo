# Apollo ERP RAG

Private Apollo manual/source retrieval using **Chroma + existing Ollama**.
Chroma stores explicit E5 vectors; Ollama provides embeddings and streamed Qwen
answers. The app validates the portable corpus and maps results back to its full
passages, chunk IDs, paths, and line/page ranges. No model training is involved.

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

Start with the 388 manual passages and **one question**:

```bash
docker compose run --rm assistant index --corpus /corpus --collection apollo-manuals --kind manual
docker compose run --rm assistant search "zasto nece da mi knjizi dokument, nije mi u ravnotezi, nije mi nasao konto zaliha za artikal 56?" --corpus /corpus --collection apollo-manuals --kind manual
docker compose run --rm assistant ask "zasto nece da mi knjizi dokument, nije mi u ravnotezi, nije mi nasao konto zaliha za artikal 56?" --corpus /corpus --collection apollo-manuals --kind manual
```

Check the retrieved passages and answer citations against the manual. The stock
account evidence is in guide lines 443-459: LIN001M001 provides warehouse/item-type
defaults and LIT002M001 an item override. Balance status is separate evidence in
lines 2252-2268. Retrieval of these passages still needs server verification. The answer
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

The accepted v3 bundle is transferred as data only; it needs no preparation for
first use. After initial v3 setup, prepare new inputs on the Ubuntu server. Put
future raw input files in the already ignored `data/inputs/` directory, then run
these commands from the application folder:

```bash
mkdir -p data/inputs data/private .cache/huggingface
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

For the new bundle, set `APOLLO_CORPUS_DIR=./data/private/corpus-v4` in `.env` and
index it into a new collection:

```bash
docker compose run --rm assistant index --corpus /corpus --collection apollo-manuals-v4 --kind manual
```

No additional Compose services or configuration are needed. Read the prepared
bundle's text for a quality check; no preparation test command is required.

## Publishing the application

From the repository root, stage only the application files listed below. Review
the staged diff before committing and pushing yourself. Upstream submodules,
`Network-Intelligence-Chat`, private data, and research documents are separate.

```bash
git status --short
git add -- nanogpt-apollo/.gitignore nanogpt-apollo/.dockerignore nanogpt-apollo/.python-version nanogpt-apollo/.env.example nanogpt-apollo/Dockerfile nanogpt-apollo/compose.yaml nanogpt-apollo/pyproject.toml nanogpt-apollo/uv.lock nanogpt-apollo/README.md nanogpt-apollo/apollo_rag nanogpt-apollo/tests nanogpt-apollo/docs/verification.md
git diff --cached --stat
git diff --cached
```

No Git stage, commit, push, server connection, model pull, or service startup was
performed during this migration. Current evidence and runtime limits are in
[docs/verification.md](docs/verification.md).
