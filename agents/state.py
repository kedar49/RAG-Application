"""
LangGraph state definition for the Multi-Agent RAG pipeline.
The State object flows through every agent node in the graph,
accumulating data as the pipeline progresses.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
import operator


class RAGState(TypedDict, total=False):
    """
    Shared state that flows through the LangGraph pipeline.
    Each agent reads what it needs and writes back its outputs.
    
    total=False makes all keys optional so each agent only sets
    the fields it produces (partial updates via operator.add or direct assign).
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    query: str
    """The original user question."""

    # ── Query Agent outputs ────────────────────────────────────────────────────
    rewritten_queries: list[str]
    """Multiple reformulations for multi-query retrieval."""

    hypothetical_answer: str
    """HyDE hypothetical answer for embedding-based retrieval."""

    query_intent: str
    """Detected intent: factual | comparison | summary | explanation."""

    needs_retrieval: bool
    """False if query is a greeting/meta-question — skip retrieval."""

    # ── Retrieval Agent outputs ────────────────────────────────────────────────
    retrieved_docs: list[Document]
    """Top-K docs after hybrid search + cross-encoder reranking."""

    retrieval_context: str
    """Retrieved docs formatted as a single context string for the LLM."""

    retrieval_debug: list[dict[str, Any]]
    """Per-document retrieval diagnostics for dense, sparse, and fused ranking."""

    # ── Validation Agent outputs ───────────────────────────────────────────────
    validation_passed: bool
    """True if retrieved context is sufficient to answer the question."""

    faithfulness_score: float
    """Estimated faithfulness score (0.0 - 1.0)."""

    validation_reasoning: str
    """Explanation from the validation agent."""

    # ── Generation Agent outputs ───────────────────────────────────────────────
    answer: str
    """The final answer with inline [Source N] citations."""

    citations: list[dict]
    """Structured source citations for UI rendering."""

    confidence: float
    """Model's confidence in the answer (0.0 - 1.0)."""

    # ── Control flow ───────────────────────────────────────────────────────────
    refused: bool
    """True if the system refuses to answer due to insufficient context."""

    refusal_reason: Optional[str]
    """Human-readable reason for refusal."""

    error: Optional[str]
    """Error message if any agent failed."""

    # ── Conversation memory ────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], operator.add]
    """Full conversation history (append-only via operator.add)."""
