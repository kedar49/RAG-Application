# RAGit

A multi-agent RAG system that refuses to hallucinate. Built with LangGraph, Ollama, and pgvector — runs entirely on your machine, no API keys needed.

---

Most RAG systems have one real problem: when the retrieved documents don't contain a good answer, the LLM makes one up anyway. RAGit is designed around solving that specific failure mode. There's a dedicated validation agent that scores the retrieved context before generation even starts. If the score is below 0.60, the system refuses and explains why — instead of confidently writing something wrong.

---

## What's inside

Four agents wired together in a LangGraph graph:

**Query Agent** — takes the raw user question and does three things before retrieval. It generates a hypothetical answer (HyDE) that gets embedded alongside the original question, which improves semantic alignment significantly. It also writes three reformulations of the query for broader retrieval coverage, and classifies the intent so greetings and meta-questions skip the knowledge base entirely.

**Retrieval Agent** — runs hybrid search over all the query variants. Dense search via pgvector cosine similarity catches semantic matches; BM25 over the candidates catches exact term matches. The two ranked lists get merged with Reciprocal Rank Fusion, then a cross-encoder reranks the final candidates. Typical flow is 20–60 candidates down to 5 that actually get sent to the LLM.

**Validation Agent** — checks whether the retrieved context is actually sufficient to answer the question. Deterministic temperature (0.0), strict prompt. If the faithfulness score comes back below threshold, the orchestrator routes to a refusal instead of generation. This is the thing that makes the system not hallucinate.

**Generation Agent** — writes the answer using only the labeled `[Source N]` chunks. Every factual claim has to cite a source inline. The JSON citation block at the end gets parsed and used by the UI to show users exactly which document and page number each claim comes from.

---

## Models

Everything runs locally through Ollama. No API keys, no data leaving your machine.

| Role | Model | Size |
|---|---|---|
| Generation | `qwen2.5:7b` | ~4.7 GB |
| Validation + Query | `phi4-mini:3.8b` | ~2.5 GB |
| Embedding | `nomic-embed-text` | ~274 MB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB |

If you're on a machine with less than 6GB VRAM, use `gemma3:4b` instead of `qwen2.5:7b`. Set `GENERATION_MODEL=gemma3:4b` in `.env`.

---

## Quick start

You need Docker, Ollama, and Python 3.11+. For detailed platform-specific instructions, see [setup.md](setup.md).

```bash
# Clone and enter the project
git clone https://github.com/kedar49/RAG-Application.git
cd RAG-Application

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env

# Start the database
docker compose up -d

# Pull models (this takes a while the first time)
ollama pull qwen2.5:7b
ollama pull phi4-mini:3.8b
ollama pull nomic-embed-text

# Run
streamlit run app.py
```

Open `http://localhost:8501`. Upload a PDF or paste a URL, then ask questions.

---

## Configuration

All tuneable settings are in `.env`. The defaults are reasonable but here are the ones worth knowing about:

```bash
# Swap generation model if VRAM is limited
GENERATION_MODEL=qwen2.5:7b     # or gemma3:4b, qwen2.5:3b

# Faithfulness threshold — below this the system refuses to answer
FAITHFULNESS_THRESHOLD=0.60     # raise to be more conservative

# How many documents to retrieve before reranking
RETRIEVAL_TOP_K=20              # wider = better recall, slower

# How many to keep after reranking
RERANK_TOP_K=5                  # what actually gets passed to the LLM

# Generation temperature — keep this low
GENERATION_TEMPERATURE=0.1      # higher = more creative = more hallucinations
```

---

## Project layout

```
ragit/
├── agents/
│   ├── orchestrator.py      # LangGraph graph definition
│   ├── query_agent.py       # HyDE + multi-query + intent detection
│   ├── retrieval_agent.py   # Hybrid search + cross-encoder reranking
│   ├── validation_agent.py  # Faithfulness gate
│   ├── generation_agent.py  # Structured answer + citations
│   └── state.py             # Shared TypedDict state
├── core/
│   ├── embedder.py          # nomic-embed-text singleton
│   ├── vector_store.py      # pgvector + BM25 + RRF fusion
│   ├── reranker.py          # CrossEncoder (ms-marco)
│   └── chunker.py           # RecursiveCharacterTextSplitter
├── ingestion/
│   ├── pdf_ingester.py      # PDF → chunks → embed → store
│   └── web_ingester.py      # URL → scrape → chunks → store
├── schemas/
│   └── __init__.py          # Pydantic response schemas
├── app.py                   # Streamlit UI
├── config.py                # Pydantic Settings
├── docker-compose.yml       # pgvector container
├── requirements.txt
├── technical.md             # Architecture deep-dive
└── setup.md                 # Platform setup (Linux/macOS/Windows)
```

---

## How ingestion works

**PDF:** Upload a file through the sidebar. It gets loaded with PyPDFLoader, split into ~512-character chunks with 50-character overlap (preserving paragraph and sentence boundaries), embedded with `nomic-embed-text`, and stored in pgvector. Each chunk gets a stable ID derived from a SHA-256 hash of its content, so ingesting the same file twice doesn't create duplicates.

**Web/URL:** Paste a URL. The scraper fetches the page, strips nav/header/footer/scripts, and extracts the readable text from `<main>` or `<article>` elements when available. Same chunking and storage process as PDF.

---

## Docs

- [technical.md](technical.md) — full architecture explanation, agent internals, model selection rationale, data flow
- [setup.md](setup.md) — step-by-step setup for Fedora/Linux, macOS, and Windows

---

## Tech stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — multi-agent graph orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM abstractions, document loaders
- [Ollama](https://ollama.com) — local model serving
- [pgvector](https://github.com/pgvector/pgvector) — vector similarity search on Postgres
- [sentence-transformers](https://www.sbert.net/) — cross-encoder reranking
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — sparse keyword search
- [Streamlit](https://streamlit.io) — UI
- [Pydantic](https://docs.pydantic.dev/) — config and schema validation
