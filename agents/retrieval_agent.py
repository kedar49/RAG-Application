"""
Retrieval Agent.

Orchestrates the full retrieval pipeline:
  1. Multi-query hybrid search (dense + BM25) for each rewritten query
  2. Result deduplication and merging
  3. Cross-encoder reranking of the merged candidates
  4. Context string construction with source attribution

The agent also embeds the HyDE hypothetical answer as an additional
query variant — this is the core of the HyDE technique.
"""

import logging
from langchain_core.documents import Document

from agents.state import RAGState
from core.vector_store import hybrid_search
from core.reranker import rerank_documents
from config import settings

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 8000   # ~2000 tokens — safe budget for context window


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicates while preserving the richest retrieval metadata."""
    seen: dict[str, Document] = {}
    for doc in docs:
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:64])
        if doc_id not in seen:
            seen[doc_id] = doc
            continue

        existing = seen[doc_id]
        for key, value in doc.metadata.items():
            if key not in existing.metadata:
                existing.metadata[key] = value
    return list(seen.values())


def _build_retrieval_debug(docs: list[Document]) -> list[dict]:
    """Extracts retrieval diagnostics for UI and evaluation."""
    debug_rows: list[dict] = []
    for i, doc in enumerate(docs, start=1):
        debug_rows.append({
            "source_number": i,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "dense_rank": doc.metadata.get("dense_rank"),
            "dense_score": doc.metadata.get("dense_score"),
            "sparse_rank": doc.metadata.get("sparse_rank"),
            "sparse_score": doc.metadata.get("sparse_score"),
            "reranker_score": doc.metadata.get("reranker_score"),
            "rrf_score": doc.metadata.get("rrf_score"),
            "retrieval_channels": doc.metadata.get("retrieval_channels", []),
        })
    return debug_rows


def _format_context(docs: list[Document]) -> str:
    """
    Builds a structured context string for the LLM.
    Each source chunk is labelled [Source N] for citation tracking.
    """
    parts: list[str] = []
    total_chars = 0

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        page_str = f" (Page {page})" if page else ""

        chunk_text = f"[Source {i}] {source}{page_str}\n{doc.page_content}"
        chunk_chars = len(chunk_text)

        if total_chars + chunk_chars > MAX_CONTEXT_CHARS:
            logger.debug(f"Context budget reached at source {i}, stopping.")
            break

        parts.append(chunk_text)
        total_chars += chunk_chars

    return "\n\n---\n\n".join(parts)


def retrieval_agent_node(state: RAGState) -> RAGState:
    """
    LangGraph node: hybrid retrieval + cross-encoder reranking.
    
    Reads:  state["query"], state["rewritten_queries"], state["hypothetical_answer"]
    Writes: state["retrieved_docs"], state["retrieval_context"]
    """
    query = state.get("query", "")
    rewritten_queries = state.get("rewritten_queries", [query])
    hypothetical_answer = state.get("hypothetical_answer", query)

    logger.info(
        f"[RetrievalAgent] Running {len(rewritten_queries)} query variants "
        f"+ 1 HyDE query"
    )

    # ── Step 1: Multi-query hybrid search ─────────────────────────────────────
    all_candidates: list[Document] = []

    # Search with each rewritten query
    for q in rewritten_queries:
        candidates = hybrid_search(q, top_k=settings.retrieval_top_k)
        all_candidates.extend(candidates)

    # Also search using the HyDE hypothetical answer (key HyDE step)
    if hypothetical_answer and hypothetical_answer not in rewritten_queries:
        hyde_candidates = hybrid_search(hypothetical_answer, top_k=10)
        all_candidates.extend(hyde_candidates)

    # ── Step 2: Deduplicate ────────────────────────────────────────────────────
    unique_candidates = _deduplicate(all_candidates)
    logger.info(
        f"[RetrievalAgent] candidates after dedup: {len(unique_candidates)}"
    )

    if not unique_candidates:
        logger.warning("[RetrievalAgent] No documents retrieved.")
        return {
            **state,
            "retrieved_docs": [],
            "retrieval_context": "",
            "retrieval_debug": [],
        }

    # ── Step 3: Cross-encoder reranking ───────────────────────────────────────
    reranked = rerank_documents(
        query=query,
        documents=unique_candidates,
        top_k=settings.rerank_top_k,
    )
    logger.info(f"[RetrievalAgent] After reranking: {len(reranked)} docs")

    # ── Step 4: Format context ────────────────────────────────────────────────
    context = _format_context(reranked)

    return {
        **state,
        "retrieved_docs": reranked,
        "retrieval_context": context,
        "retrieval_debug": _build_retrieval_debug(reranked),
    }
