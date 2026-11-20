# Apollo ERP RAG

Apollo ERP RAG is a local support assistant for Apollo forms, documents, and
workflows. It answers in Serbian Latin and cites the original passage,
manual, and line range. The app reads a prepared corpus, searches Chroma, and
calls a local Ollama model through Docker Compose.

## Start here

- [Application and Ubuntu setup](nanogpt-apollo/README.md)
- [Retrieval and answer evaluation](nanogpt-apollo/docs/serbian-rag-evaluation.md#compare-routed-retrieval-with-the-hybrid-baseline)
- [Benchmark questions and expected evidence locations](nanogpt-apollo/docs/serbian-support-cases.json)

## How the app works

1. Preparation reads UTF-8 manuals and eligible source files, splits them into
   passages, and records citations and provenance.
2. Indexing embeds those passages with multilingual E5 and stores the vectors
   in Chroma. The search command prints ranked passages without generation.
3. With the experimental routed ranker, search handles form codes as exact
   lookups, combines keyword and vector search for other questions, and
   reserves evidence for distinct topics.
4. Qwen 3.5 4B in Ollama uses the selected evidence to answer. Evaluation
   records retrieval coverage and the generated answer separately.

Start a code review in [the CLI](nanogpt-apollo/apollo_rag/cli.py),
[retrieval](nanogpt-apollo/apollo_rag/retrieval.py),
[answer generation](nanogpt-apollo/apollo_rag/answers.py), and
[evaluation](nanogpt-apollo/apollo_rag/evaluation.py).

Bring your own authorized documents. Private manuals, source exports, corpus
bundles, indexes, and generated reports are stored separately. The assistant
does not connect to live Oracle or run ERP actions. The upstream training
submodules are reference experiments and are optional for the RAG app.
