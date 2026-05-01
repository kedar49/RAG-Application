"""
Hybrid vector store: dense (pgvector) + sparse (BM25) search.

Two retrieval strategies combined via Reciprocal Rank Fusion (RRF):
  1. Dense:  pgvector cosine similarity — captures semantic meaning
  2. Sparse: BM25 keyword ranking — captures exact term matches

Why hybrid matters:
- Dense-only misses exact entity/keyword matches (e.g., "RFC 1234", "Article 5.3.2")
- Sparse-only misses semantic similarity (e.g., "auto" vs "car")
- Hybrid consistently outperforms either alone on BEIR benchmark

RRF formula: RRF(d) = Σ 1 / (k + rank_i(d))  [k=60 by default]
"""

import logging
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.embedder import get_embedder_singleton
from config import settings

logger = logging.getLogger(__name__)


def get_vector_store() -> PGVector:
    """Returns a configured PGVector store connected to pgvector."""
    return PGVector(
        connection_string=settings.db_url,
        collection_name=settings.collection_name,
        embedding_function=get_embedder_singleton(),
        use_jsonb=True,  # JSONB metadata storage for efficient filtering
    )


def _reciprocal_rank_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion.
    Returns a deduplicated, fused ranking.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:64])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(sparse_docs):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:64])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        doc_map[doc_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


def hybrid_search(
    query: str,
    top_k: int | None = None,
) -> list[Document]:
    """
    Performs hybrid search: dense pgvector + in-memory BM25 on the dense results.
    
    Strategy:
    1. Fetch top_k * 2 candidates from pgvector (dense)
    2. Re-rank those candidates with BM25 (sparse) over their text
    3. Fuse with RRF → top_k documents
    
    Args:
        query:  The search query (raw or rewritten).
        top_k:  Number of documents to return. Defaults to settings.retrieval_top_k.
    
    Returns:
        List of Documents sorted by fused RRF score.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    vs = get_vector_store()

    # ── Step 1: Dense retrieval ────────────────────────────────────────────────
    try:
        dense_docs = vs.similarity_search(query, k=top_k)
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")
        return []

    if not dense_docs:
        return []

    # ── Step 2: BM25 on the dense candidate set ────────────────────────────────
    tokenized_corpus = [doc.page_content.lower().split() for doc in dense_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    sparse_docs = sorted(
        zip(bm25_scores, dense_docs),
        key=lambda x: x[0],
        reverse=True,
    )
    sparse_ranked = [doc for _, doc in sparse_docs]

    # ── Step 3: RRF fusion ─────────────────────────────────────────────────────
    fused = _reciprocal_rank_fusion(dense_docs, sparse_ranked)

    logger.debug(
        f"Hybrid search: dense={len(dense_docs)}, fused={len(fused)}, top_k={top_k}"
    )
    return fused[:top_k]


def add_documents(documents: list[Document]) -> list[str]:
    """
    Adds pre-chunked documents to the vector store.
    Uses chunk_id as the document ID for upsert/deduplication.

    Returns:
        List of document IDs that were added.
    """
    vs = get_vector_store()
    ids = [doc.metadata.get("chunk_id", "") for doc in documents]
    vs.add_documents(documents, ids=ids)
    logger.info(f"Added {len(documents)} chunks to vector store.")
    return ids


def clear_collection() -> None:
    """Deletes all documents from the vector store collection."""
    vs = get_vector_store()
    vs.delete_collection()
    logger.info(f"Cleared collection: {settings.collection_name}")
