"""
Centralized configuration for the Multi-Agent RAG system.
All values can be overridden via environment variables or a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ───────────────────────────────────────────────────────────────
    db_url: str = "postgresql+psycopg://ai:ai@localhost:5532/ai"
    collection_name: str = "rag_documents_v2"

    # ── Ollama ─────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # ── Models ─────────────────────────────────────────────────────────────────
    # qwen2.5:7b → best RAG quality at 7B class (recommended)
    # gemma3:4b  → lighter, 128K ctx (use if VRAM < 6GB)
    # qwen2.5:3b → smallest option (use if VRAM < 4GB)
    generation_model: str = "qwen2.5:7b"

    # phi4-mini:3.8b → Microsoft Phi-4 Mini, best reasoning at small scale
    validation_model: str = "phi4-mini:3.8b"

    # nomic-embed-text → dedicated embedding model, 768 dims
    # NEVER use the chat model as the embedder
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieval_top_k: int = 20   # dense + sparse, before reranking
    rerank_top_k: int = 5       # final docs passed to generation agent

    # ── Anti-Hallucination ─────────────────────────────────────────────────────
    faithfulness_threshold: float = 0.60  # below this → refuse to answer
    generation_temperature: float = 0.1   # keep low for factual RAG

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50


settings = Settings()
