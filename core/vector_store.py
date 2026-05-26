"""
Hybrid vector store: dense pgvector + corpus-level sparse BM25 retrieval.

Dense retrieval handles semantic similarity while sparse retrieval catches
exact-term matches that embeddings often miss. The two ranked lists are fused
with Reciprocal Rank Fusion (RRF), and each returned document is annotated
with retrieval diagnostics for downstream inspection.
"""

import logging
import re
from typing import Any

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.embedder import get_embedder_singleton
from config import settings

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\b\w+\b")

_SPARSE_CACHE: dict[str, Any] = {
    "collection": None,
    "documents": [],
    "tokenized_corpus": [],
    "bm25": None,
}


def get_vector_store() -> PGVector:
    """Returns a configured PGVector store connected to pgvector."""
    return PGVector(
        connection_string=settings.db_url,
        collection_name=settings.collection_name,
        embedding_function=get_embedder_singleton(),
        use_jsonb=True,
    )


def _tokenize(text: str) -> list[str]:
    """Simple normalization for sparse retrieval."""
    return TOKEN_PATTERN.findall(text.lower())


def _invalidate_sparse_cache() -> None:
    """Clears the cached sparse corpus after ingestion or deletion."""
    _SPARSE_CACHE.update({
        "collection": None,
        "documents": [],
        "tokenized_corpus": [],
        "bm25": None,
    })


def _load_collection_documents(
    vs: PGVector,
    filter: dict | None = None,
) -> list[Document]:
    """Loads every document in the current collection for sparse retrieval."""
    with vs._make_session() as session:
        collection = vs.get_collection(session)
        if not collection:
            return []

        query = session.query(vs.EmbeddingStore).filter(
            vs.EmbeddingStore.collection_id == collection.uuid
        )

        if filter:
            filter_clause = vs._create_filter_clause(filter)
            if filter_clause is not None:
                query = query.filter(filter_clause)

        rows = query.all()

    documents: list[Document] = []
    for row in rows:
        documents.append(
            Document(
                page_content=row.document,
                metadata=dict(row.cmetadata or {}),
            )
        )

    return documents


def list_sources() -> list[dict[str, Any]]:
    """Returns basic source inventory for the current collection."""
    vs = get_vector_store()
    docs = _load_collection_documents(vs)
    sources: dict[str, dict[str, Any]] = {}

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        row = sources.setdefault(source, {
            "source": source,
            "source_type": doc.metadata.get("source_type"),
            "chunks": 0,
            "pages": set(),
            "url": doc.metadata.get("url"),
            "title": doc.metadata.get("source_title"),
        })
        row["chunks"] += 1
        if doc.metadata.get("page") is not None:
            row["pages"].add(doc.metadata["page"])

    inventory: list[dict[str, Any]] = []
    for row in sources.values():
        inventory.append({
            "source": row["source"],
            "source_type": row["source_type"],
            "chunks": row["chunks"],
            "pages": len(row["pages"]),
            "url": row["url"],
            "title": row["title"],
        })

    inventory.sort(key=lambda item: (item["source_type"] or "", item["source"]))
    return inventory


def source_exists(source_name: str) -> bool:
    """Checks whether a source with the given display name already exists."""
    return any(row["source"] == source_name for row in list_sources())


def _get_sparse_corpus(vs: PGVector, filter: dict | None = None) -> tuple[list[Document], BM25Okapi | None]:
    """
    Returns the cached sparse corpus.

    Filtering is currently uncached because filtered corpora can vary widely.
    The default unfiltered corpus is cached and invalidated on writes.
    """
    if filter:
        docs = _load_collection_documents(vs, filter=filter)
        tokenized = [_tokenize(doc.page_content) for doc in docs]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        return docs, bm25

    if (
        _SPARSE_CACHE["collection"] == settings.collection_name
        and _SPARSE_CACHE["bm25"] is not None
    ):
        return _SPARSE_CACHE["documents"], _SPARSE_CACHE["bm25"]

    docs = _load_collection_documents(vs)
    tokenized = [_tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    _SPARSE_CACHE.update({
        "collection": settings.collection_name,
        "documents": docs,
        "tokenized_corpus": tokenized,
        "bm25": bm25,
    })
    return docs, bm25


def _annotate_dense_results(
    dense_results: list[tuple[Document, float]],
) -> list[Document]:
    """Adds dense retrieval diagnostics to documents."""
    annotated: list[Document] = []
    for rank, (doc, score) in enumerate(dense_results, start=1):
        doc.metadata["dense_rank"] = rank
        doc.metadata["dense_score"] = round(float(score), 6)
        annotated.append(doc)
    return annotated


def _rank_sparse_documents(
    query: str,
    corpus_docs: list[Document],
    bm25: BM25Okapi | None,
    top_k: int,
) -> list[Document]:
    """Runs BM25 over the full corpus and annotates the top results."""
    if not corpus_docs or bm25 is None:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    scored_docs = sorted(
        zip(scores, corpus_docs),
        key=lambda item: item[0],
        reverse=True,
    )

    ranked: list[Document] = []
    for rank, (score, doc) in enumerate(scored_docs[:top_k], start=1):
        metadata = dict(doc.metadata)
        metadata["sparse_rank"] = rank
        metadata["sparse_score"] = round(float(score), 6)
        ranked.append(Document(page_content=doc.page_content, metadata=metadata))

    return ranked


def _reciprocal_rank_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Merges dense and sparse rankings using Reciprocal Rank Fusion.

    The returned document metadata includes:
    - dense_rank / dense_score
    - sparse_rank / sparse_score
    - rrf_score
    - retrieval_channels
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:64])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = Document(
                page_content=doc.page_content,
                metadata=dict(doc.metadata),
            )
        else:
            doc_map[doc_id].metadata.update(doc.metadata)

    for rank, doc in enumerate(sparse_docs, start=1):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:64])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = Document(
                page_content=doc.page_content,
                metadata=dict(doc.metadata),
            )
        else:
            doc_map[doc_id].metadata.update(doc.metadata)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    fused_docs: list[Document] = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        channels: list[str] = []
        if "dense_rank" in doc.metadata:
            channels.append("dense")
        if "sparse_rank" in doc.metadata:
            channels.append("sparse")
        doc.metadata["rrf_score"] = round(scores[doc_id], 6)
        doc.metadata["retrieval_channels"] = channels
        fused_docs.append(doc)

    return fused_docs


def hybrid_search(
    query: str,
    top_k: int | None = None,
    filter: dict | None = None,
) -> list[Document]:
    """
    Performs true hybrid retrieval:
    1. Dense search over pgvector
    2. Sparse BM25 search over the collection corpus
    3. Reciprocal Rank Fusion over both rankings
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    vs = get_vector_store()

    try:
        dense_results = vs.similarity_search_with_score(
            query,
            k=max(top_k, settings.dense_search_k),
            filter=filter,
        )
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")
        return []

    dense_docs = _annotate_dense_results(dense_results)
    corpus_docs, bm25 = _get_sparse_corpus(vs, filter=filter)
    sparse_docs = _rank_sparse_documents(
        query=query,
        corpus_docs=corpus_docs,
        bm25=bm25,
        top_k=max(top_k, settings.sparse_search_k),
    )

    if not dense_docs and not sparse_docs:
        return []

    fused = _reciprocal_rank_fusion(dense_docs, sparse_docs)
    logger.debug(
        "Hybrid search: dense=%s sparse=%s fused=%s top_k=%s",
        len(dense_docs),
        len(sparse_docs),
        len(fused),
        top_k,
    )
    return fused[:top_k]


def add_documents(documents: list[Document]) -> list[str]:
    """
    Adds pre-chunked documents to the vector store.
    Uses chunk_id as the document ID for upsert/deduplication.
    """
    vs = get_vector_store()
    ids = [doc.metadata.get("chunk_id", "") for doc in documents]
    vs.add_documents(documents, ids=ids)
    _invalidate_sparse_cache()
    logger.info(f"Added {len(documents)} chunks to vector store.")
    return ids


def clear_collection() -> None:
    """Deletes all documents from the vector store collection."""
    vs = get_vector_store()
    vs.delete_collection()
    _invalidate_sparse_cache()
    logger.info(f"Cleared collection: {settings.collection_name}")
