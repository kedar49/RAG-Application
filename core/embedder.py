"""
Embedding module — always uses a dedicated embedding model.

CRITICAL: Never use a chat/generation model for embeddings.
nomic-embed-text is a purpose-built embedding model that produces
much higher quality vectors than using llama or qwen for embedding.
"""

from langchain_ollama import OllamaEmbeddings
from config import settings


def get_embedder() -> OllamaEmbeddings:
    """
    Returns a configured OllamaEmbeddings instance using nomic-embed-text.
    
    nomic-embed-text:
    - 768-dimensional embeddings
    - Trained specifically for semantic similarity tasks
    - ~274MB download via `ollama pull nomic-embed-text`
    - Consistently outperforms using chat models as embedders
    """
    return OllamaEmbeddings(
        model=settings.embedding_model,      # nomic-embed-text
        base_url=settings.ollama_base_url,
    )


# Module-level singleton — embedder is stateless, safe to reuse
_embedder: OllamaEmbeddings | None = None


def get_embedder_singleton() -> OllamaEmbeddings:
    """
    Returns a singleton embedder instance.
    Avoids re-initializing the embedder on every call.
    """
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder
