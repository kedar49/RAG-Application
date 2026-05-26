# GroundedRAG

A local multi-agent RAG system built with LangGraph, Ollama, and pgvector. It is designed to stay grounded, refuse weak answers, and show its retrieval trail instead of guessing.

## What it does

- Rewrites queries with HyDE plus multi-query expansion
- Routes greetings and app-help prompts to a dedicated meta agent
- Runs true hybrid retrieval: dense pgvector search + corpus-level BM25 + RRF fusion
- Reranks results with a cross-encoder
- Validates whether the retrieved context is sufficient before generation
- Rejects answers that are not properly grounded in retrieved sources
- Shows citations, retrieval diagnostics, and source inventory in the UI

## Architecture

The current graph has five nodes:

- `query_agent`: analyzes the question, rewrites it, and decides whether retrieval is needed
- `meta_agent`: handles greetings and product-help questions without pretending to cite documents
- `retrieval_agent`: runs hybrid retrieval across rewritten queries and HyDE output
- `validation_agent`: checks whether the retrieved context is sufficient
- `generation_agent`: produces a cited answer, or refuses if grounding checks fail

If validation fails, the graph routes to a deterministic refusal response instead of generation.

## Retrieval stack

Retrieval is now a real hybrid pipeline:

1. Dense search over pgvector with `nomic-embed-text`
2. Sparse BM25 ranking over the indexed corpus
3. Reciprocal Rank Fusion (RRF) to merge dense and sparse rankings
4. Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`

Each returned chunk is annotated with diagnostics such as:

- `dense_rank`
- `dense_score`
- `sparse_rank`
- `sparse_score`
- `rrf_score`
- `reranker_score`
- `retrieval_channels`

These are surfaced in the Streamlit UI under `Pipeline Details`.

## Ingestion

### PDF

- Extracts text with `PyPDFLoader`
- Captures PDF metadata such as title, author, and total page count when available
- Chunks with token-aware splitting
- Stores stable chunk IDs that include page/document context plus content hash

### Web / URL

- Fetches the page with `requests`
- Removes noisy elements like nav, header, footer, scripts, and forms
- Extracts section-aware documents using heading structure (`h1`/`h2`/`h3`)
- Stores section metadata such as `source_title`, `section_heading`, and `section_path`

Duplicate sources are skipped instead of being re-ingested.

## UI

The Streamlit app includes:

- Chat interface with grounded answers and refusal handling
- Source inventory in the sidebar
- Retrieval diagnostics table in pipeline details
- Inline source citations and excerpts
- Knowledge base clear/reset controls

If the database is unavailable, the sidebar now shows that source inventory cannot be loaded.

## Quick start

You need Docker, Ollama, and Python 3.11+.

```bash
git clone https://github.com/kedar49/RAG-Application.git
cd RAG-Application

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

docker compose up -d

ollama pull qwen2.5:7b
ollama pull phi4-mini:3.8b
ollama pull nomic-embed-text

streamlit run app.py
```

Open `http://localhost:8501`.

## Important settings

```bash
GENERATION_MODEL=qwen2.5:7b
VALIDATION_MODEL=phi4-mini:3.8b
EMBEDDING_MODEL=nomic-embed-text

RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
DENSE_SEARCH_K=40
SPARSE_SEARCH_K=40

FAITHFULNESS_THRESHOLD=0.60
GENERATION_TEMPERATURE=0.1

CHUNK_SIZE=512
CHUNK_OVERLAP=50
CHUNK_TOKEN_ENCODING=cl100k_base
```

## Project layout

```text
agents/
  generation_agent.py
  meta_agent.py
  orchestrator.py
  query_agent.py
  retrieval_agent.py
  state.py
  validation_agent.py
core/
  chunker.py
  embedder.py
  reranker.py
  vector_store.py
ingestion/
  pdf_ingester.py
  web_ingester.py
schemas/
  __init__.py
app.py
config.py
docker-compose.yml
README.md
setup.md
technical.md
```

## Notes

- The current code still depends on the local pgvector database being available at `localhost:5532`.
- The repo compiles cleanly, but retrieval and ingestion need the DB running to be smoke-tested end to end.
- `langchain_community.vectorstores.PGVector` is deprecated upstream, so a future migration to `langchain_postgres` is a good next step.

## Docs

- [technical.md](technical.md)
- [setup.md](setup.md)
