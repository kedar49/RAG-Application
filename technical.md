# Technical Architecture

RAGit is a multi-agent RAG pipeline built on LangGraph. The core design goal was simple: never hallucinate, even if it means refusing to answer. Everything in this codebase serves that one constraint.

---

## How the pipeline fits together

A user query enters the system and flows through four agents in a directed graph. Each agent reads from a shared state object, does its work, and writes its outputs back. The orchestrator decides which agent runs next based on what the previous one returned.

```
User Query
    │
    ▼
[query_agent]          ← rewrites the query, classifies intent
    │
    ├── greeting/meta? → [generation_agent] → END
    │
    ▼
[retrieval_agent]      ← hybrid search + cross-encoder reranking
    │
    ▼
[validation_agent]     ← faithfulness gate
    │
    ├── score < 0.60? → [refusal_node] → END
    │
    ▼
[generation_agent]     ← structured answer with [Source N] citations
    │
    ▼
   END
```

The graph is built once at startup by `build_rag_graph()` in `orchestrator.py` and reused across requests. There's no re-compilation overhead per query.

---

## Shared State (`agents/state.py`)

Every agent reads and writes a single `RAGState` TypedDict that flows through the graph. Because it's `total=False`, agents only set the keys they actually produce — no agent needs to touch fields outside its scope.

```python
class RAGState(TypedDict, total=False):
    # Input
    query: str

    # Query Agent outputs
    rewritten_queries: list[str]
    hypothetical_answer: str
    query_intent: str
    needs_retrieval: bool

    # Retrieval Agent outputs
    retrieved_docs: list[Document]
    retrieval_context: str

    # Validation Agent outputs
    validation_passed: bool
    faithfulness_score: float
    validation_reasoning: str

    # Generation Agent outputs
    answer: str
    citations: list[dict]
    confidence: float

    # Control flow
    refused: bool
    refusal_reason: Optional[str]
    error: Optional[str]

    # Conversation memory (append-only)
    messages: Annotated[list[BaseMessage], operator.add]
```

The `messages` field uses `operator.add` as its reducer — LangGraph will append new messages rather than overwrite the list, which is what you want for conversation history.

---

## Agent breakdown

### 1. Query Agent (`agents/query_agent.py`)

**Model:** `phi4-mini:3.8b` at temperature 0.3

The query agent does three things before any retrieval happens:

**HyDE (Hypothetical Document Embeddings):** It writes a plausible answer to the question, even if it doesn't actually know the answer. That hypothetical answer gets embedded and used as an additional search query. The intuition: a hypothetical answer lives in a much better part of the embedding space than a short, vague user question does.

For example: `"what does the report say about Q3?"` is hard to match by embedding alone. But a generated paragraph like `"In Q3, revenue increased by 12% driven by..."` will find the right chunks much more reliably.

**Multi-query expansion:** Three reformulations of the original query, each phrased differently. The retrieval agent runs all of them and merges the results. This handles cases where the vocabulary in the document doesn't match what the user typed.

**Intent classification:** `factual | comparison | summary | explanation | other`. Currently used for routing — greetings and meta-questions skip retrieval entirely.

The model is forced to output JSON (`format="json"` in ChatOllama). If parsing fails for any reason, the agent falls back to the original query and continues — it never crashes the pipeline.

---

### 2. Retrieval Agent (`agents/retrieval_agent.py`)

**No LLM involved — pure retrieval logic.**

This is where the hybrid search happens. The agent runs all the rewritten queries plus the HyDE hypothetical answer through `hybrid_search()`, deduplicates the results by `chunk_id`, then reranks them with a cross-encoder.

**Why hybrid search instead of just vector search?**

Dense embeddings are great for semantic similarity (`"car"` ↔ `"automobile"`) but they miss exact term matches. If a document contains `"RFC 7519"` or `"Article 5.3.2"`, pure vector search might score that chunk poorly because those strings have no semantic neighborhood in the embedding space. BM25 catches those exact matches.

The results from dense and sparse retrieval are merged using **Reciprocal Rank Fusion (RRF)**:

```
RRF(d) = Σ 1 / (k + rank_i(d))   where k = 60
```

A document gets a higher fused score if it appears highly ranked in *both* lists. RRF doesn't require score normalization across the two systems, which is one reason it's popular for hybrid search.

**Cross-encoder reranking:** After deduplication, the top candidates go to the cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Unlike the bi-encoder used for embedding (which encodes query and document separately), a cross-encoder sees the query and document concatenated together. This joint encoding lets it model the actual interaction between them, giving much more accurate relevance scores.

Typical flow: 20–60 candidates from hybrid search → top 5 after cross-encoder reranking.

The context passed to the LLM is capped at ~8000 characters (~2000 tokens) with each chunk labeled `[Source N]` for citation tracking.

---

### 3. Validation Agent (`agents/validation_agent.py`)

**Model:** `phi4-mini:3.8b` at temperature 0.0 (deterministic)

This is the most important node in the graph. Without it, the generation LLM would hallucinate answers whenever the retrieved context doesn't contain what the user asked for — that's just how LLMs work.

The validation agent checks: *does the retrieved context actually contain information that answers this question?*

It outputs a `faithfulness_score` from 0.0 to 1.0 and an `is_sufficient` boolean. If the score falls below `FAITHFULNESS_THRESHOLD` (default 0.60), the orchestrator routes to the refusal node regardless of what `is_sufficient` says. The threshold check is in code, not left to the model — models can be overconfident.

Temperature is 0.0 here intentionally. Validation should be consistent: the same context + query pair should always produce the same judgment.

If validation fails to parse the model's response for any reason, the agent defaults to `validation_passed=False`. Conservative failure is safer than optimistic failure.

---

### 4. Generation Agent (`agents/generation_agent.py`)

**Model:** `qwen2.5:7b` at temperature 0.1

Temperature 0.1 is deliberate. RAG is a factual retrieval task, not a creative one. Higher temperature increases hallucination risk; there's no upside to it here.

The generation prompt forces the model to:
- Only use information from the labeled `[Source N]` chunks
- Cite every factual claim inline
- Emit a JSON citation block after the answer

The JSON block format:
```json
{
  "citations": [
    {"source_number": 1, "source_name": "filename.pdf", "excerpt": "..."}
  ],
  "confidence": 0.9
}
```

The `_parse_generation_response()` function splits the answer text from the JSON block using a regex, cross-references source numbers against the actual document metadata, and builds the final citation list. If the JSON block is missing or malformed, it falls back to generating citations from the retrieved documents directly.

---

### 5. Refusal Node (`agents/generation_agent.py`)

Not really an "agent" — it's a deterministic node that runs when validation fails. It writes a structured refusal message explaining what was searched for and why an answer couldn't be given, rather than a generic "I don't know."

The refusal node explicitly tells users to try rephrasing or uploading more relevant documents. This is better UX than a vague error.

---

## Core infrastructure

### Vector Store (`core/vector_store.py`)

Built on pgvector (Postgres with the vector extension). Documents are stored with JSONB metadata for efficient filtering. The collection is `rag_documents_v2`.

The hybrid search function:
1. Fetches top-K candidates via pgvector cosine similarity (dense)
2. Re-ranks those candidates in memory using BM25Okapi (sparse)
3. Fuses the two ranked lists with RRF
4. Returns the top-K fused results

Note: BM25 here runs in-memory over the dense candidates, not over the full corpus. This is intentional — fetching the full corpus for BM25 on every query would be slow. By applying BM25 to the already-retrieved dense candidates, you get the keyword-sensitivity benefit without the full BM25 index cost.

### Embedder (`core/embedder.py`)

`nomic-embed-text` via Ollama — a dedicated 768-dimension embedding model. The embedder is a singleton (loaded once per process lifetime). Never use the chat model as the embedder; the vector spaces are incompatible and retrieval quality will be garbage.

### Chunker (`core/chunker.py`)

Uses `RecursiveCharacterTextSplitter` with paragraph > sentence > word priority. Chunk size is 512 characters with 50-character overlap. Each chunk gets a stable `chunk_id` derived from a SHA-256 hash of its content, which enables upsert deduplication in the vector store — ingesting the same document twice won't create duplicates.

### Reranker (`core/reranker.py`)

`cross-encoder/ms-marco-MiniLM-L-6-v2` — about 90MB, trained on MS MARCO passage ranking. Loaded once via `lru_cache(maxsize=1)` and kept in memory. Runs on CPU in ~50ms for a batch of 20 documents. If `sentence-transformers` isn't installed, it degrades gracefully to returning the top-K dense results.

---

## Ingestion pipelines

**PDF (`ingestion/pdf_ingester.py`):** Receives raw bytes from the Streamlit uploader, writes to a temp file (PyPDFLoader needs a file path, not a stream), extracts pages, chunks them, and stores. The temp file is always deleted in a `finally` block.

**Web (`ingestion/web_ingester.py`):** Fetches the URL with a 10-second timeout, extracts readable text using BeautifulSoup (removes nav, footer, scripts, etc.), and prioritizes `<main>` or `<article>` elements over the full body. Only HTML and plain text content types are supported.

---

## Configuration (`config.py`)

All tuneable parameters live in a single `Settings` class backed by Pydantic Settings. Environment variables override defaults. A `.env` file is supported. Nothing is hardcoded in agent logic.

Key parameters and their rationale:

| Parameter | Default | Why |
|---|---|---|
| `chunk_size` | 512 | ~128 tokens — fits well within most model context windows while still being semantically coherent |
| `chunk_overlap` | 50 | Prevents losing context at chunk boundaries |
| `retrieval_top_k` | 20 | Wide recall before reranking |
| `rerank_top_k` | 5 | Narrow precision after reranking |
| `faithfulness_threshold` | 0.60 | Below this, the context is too weak to trust |
| `generation_temperature` | 0.1 | Low = deterministic = fewer hallucinations |

---

## Model choices

| Role | Model | Why this one |
|---|---|---|
| Generation | `qwen2.5:7b` | Best instruction-following at the 7B class; strong at structured JSON output |
| Validation + Query | `phi4-mini:3.8b` | Microsoft Phi-4 Mini; punches well above its weight on reasoning tasks |
| Embedding | `nomic-embed-text` | Dedicated embedding model, 768 dims, optimized for retrieval |
| Reranker | `ms-marco-MiniLM-L-6-v2` | CPU-friendly, trained on passage ranking, ~90MB |

If you're on a machine with less than 6GB VRAM, swap `qwen2.5:7b` for `gemma3:4b` — it has a 128K context window and is lighter. Change `GENERATION_MODEL` in `.env`.

---

## Data flow summary

```
PDF/URL
  → extract text
  → semantic chunk (512 chars, 50 overlap)
  → embed with nomic-embed-text
  → store in pgvector (with JSONB metadata)

Query
  → phi4-mini rewrites query (HyDE + multi-query + intent)
  → hybrid_search: pgvector cosine + BM25 on candidates → RRF fusion
  → cross-encoder reranks top-20 → top-5
  → phi4-mini validates faithfulness (score 0-1, threshold 0.60)
  → if passes: qwen2.5:7b generates answer with [Source N] citations
  → if fails: structured refusal with explanation
```
