"""
Cross-encoder reranker.

Two-stage retrieval strategy:
  1. Vector DB returns top-K candidates by approximate cosine similarity (fast, high recall)
  2. Cross-encoder reranks those candidates by true relevance score (slower, high precision)

Cross-encoders are much more accurate than bi-encoders for relevance scoring
because they jointly encode the query AND the document together, seeing their
full interaction — unlike bi-encoders that encode them separately.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~90MB download
  - Trained on MS MARCO passage ranking (26M query-passage pairs)
  - Runs on CPU in ~50ms per batch of 20 docs — fast enough for real-time use
"""

from functools import lru_cache
from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_reranker() -> "CrossEncoder":
    """
    Loads the cross-encoder model once and caches it in memory.
    lru_cache(1) ensures we never load it twice across the app lifetime.
    """
    if not RERANKER_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for reranking. "
            "Run: pip install sentence-transformers"
        )
    return CrossEncoder(RERANKER_MODEL)


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """
    Reranks a list of candidate documents using the cross-encoder.
    
    Args:
        query:     The user query (or rewritten variant).
        documents: Candidate docs from the vector store (top-20 or so).
        top_k:     How many to return after reranking (typically 5).
    
    Returns:
        Top-K documents sorted by cross-encoder relevance score (highest first).
    """
    if not documents:
        return []

    # Graceful fallback if sentence-transformers not installed
    if not RERANKER_AVAILABLE:
        return documents[:top_k]

    reranker = _load_reranker()

    # Cross-encoder expects (query, passage) pairs
    pairs = [(query, doc.page_content) for doc in documents]
    scores: list[float] = reranker.predict(pairs).tolist()

    # Attach reranker score to metadata for transparency
    for doc, score in zip(documents, scores):
        doc.metadata["reranker_score"] = round(score, 4)

    # Sort by descending score, take top_k
    scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:top_k]]

    return top_docs
