# Technical Architecture

GroundedRAG is a local-first multi-agent RAG pipeline built on LangGraph. The main goal is still the same: avoid hallucinations by refusing weak answers. The current codebase now has a more honest non-RAG path, stronger retrieval, and stricter grounding checks than the original version.

## Graph

Current graph:

```text
User Query
    |
    v
[query_agent]
    |
    +--> meta question / greeting --> [meta_agent] --------> END
    |
    v
[retrieval_agent]
    |
    v
[validation_agent]
    |
    +--> insufficient context ------> [refusal_node] ------> END
    |
    v
[generation_agent]
    |
    v
   END
```

The compiled graph is cached and reused instead of being rebuilt on every request.

## Agents

### Query agent

Responsibilities:

- generates a HyDE-style hypothetical answer
- creates multiple rewritten queries
- classifies intent
- decides whether retrieval is needed

It now receives recent chat history so follow-up questions are interpreted in context.

### Meta agent

This is the new non-retrieval path. It handles:

- greetings
- brief app-help questions
- conversational prompts that should not pretend to be document-grounded

This avoids the earlier design problem where a no-retrieval request could still flow into the grounded generation prompt.

### Retrieval agent

The retrieval path is now:

1. run hybrid retrieval for each rewritten query
2. run hybrid retrieval for the HyDE answer
3. deduplicate while preserving retrieval metadata
4. rerank with cross-encoder
5. build context for generation
6. expose retrieval diagnostics

`retrieval_debug` is now part of graph state and contains per-source values like dense rank, sparse rank, RRF score, and reranker score.

### Validation agent

Validation still acts as the anti-hallucination gate. It asks whether the retrieved context is sufficient, produces a faithfulness score, and routes weak evidence to refusal.

### Generation agent

Generation now has stricter post-checks:

- parses the answer and citation JSON block
- checks that inline `[Source N]` references exist
- checks that structured citations refer to retrieved chunks
- rejects answers that are not grounded enough

If these checks fail, the user gets a refusal-style explanation instead of a silent empty answer.

## Retrieval

### Dense retrieval

Dense retrieval uses `PGVector.similarity_search_with_score()` with the Ollama embedding model configured in `core/embedder.py`.

Metadata added during dense retrieval:

- `dense_rank`
- `dense_score`

### Sparse retrieval

Sparse retrieval now runs BM25 across the indexed collection corpus, not just the dense candidate set.

Implementation details:

- documents are loaded from the current pgvector collection
- corpus tokenization is cached in memory
- cache is invalidated on ingest and on collection clear

Metadata added during sparse retrieval:

- `sparse_rank`
- `sparse_score`

### Fusion

Dense and sparse rankings are merged with Reciprocal Rank Fusion:

```text
RRF(d) = sum(1 / (k + rank_i(d)))
```

Metadata added during fusion:

- `rrf_score`
- `retrieval_channels`

### Reranking

The reranker still uses `cross-encoder/ms-marco-MiniLM-L-6-v2` and adds:

- `reranker_score`

## Chunking

Chunking is now token-aware instead of character-count-based.

Key changes:

- chunk length uses `tiktoken`
- `chunk_size` and `chunk_overlap` now reflect token budget more honestly
- chunk text is prefixed with structural hints like title and section path
- chunk metadata includes `token_count`

Chunk IDs were also strengthened to avoid collisions. They now include:

- source name
- page marker
- source document index
- chunk index within document
- content hash

## Ingestion

### PDF ingestion

PDF ingestion now adds richer metadata:

- `source_title`
- `author` when available
- `total_pages`
- `page_number`

Duplicate PDFs are skipped based on source name.

### Web ingestion

Web ingestion is now section-aware.

Instead of flattening a page into one large blob, it:

- cleans the HTML
- finds `main`, `article`, or `body`
- tracks `h1`/`h2`/`h3` headings
- groups nearby content under heading paths
- stores `section_heading` and `section_path`

Duplicate URLs are skipped based on canonicalized source name.

## UI

The UI now exposes more system internals:

- sidebar source inventory
- duplicate-ingest feedback
- retrieval diagnostics table in pipeline details
- clearer handling when source inventory cannot be loaded because the database is unavailable

## Current limitations

- `list_sources()` and duplicate checks currently read collection documents through the vector store layer, which is simple but not the most efficient approach for very large corpora.
- validation still reasons over a truncated context window rather than doing structured per-chunk evidence scoring.
- the app still depends on the deprecated `langchain_community` PGVector implementation.
- there is still no automated evaluation harness for retrieval recall, citation accuracy, or refusal precision.

## Good next steps

1. Migrate from `langchain_community.vectorstores.PGVector` to `langchain_postgres`.
2. Add structured evals for retrieval and grounding.
3. Introduce retrieve-validate-rewrite retry loops for harder questions.
4. Add metadata filters and source-level delete/reingest controls.
