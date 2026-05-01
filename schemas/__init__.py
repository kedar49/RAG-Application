"""
Pydantic schemas for structured RAG outputs.
All agent responses are validated against these schemas to prevent
malformed or incomplete answers from reaching the user.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single grounded citation backing a claim in the answer."""
    source: str = Field(description="Document source file or URL")
    page: Optional[int] = Field(default=None, description="Page number if from a PDF")
    chunk_id: Optional[str] = Field(default=None, description="Internal chunk identifier")
    excerpt: str = Field(description="Relevant excerpt from the source document")


class RAGResponse(BaseModel):
    """
    Structured answer from the generation agent.
    Every answer MUST include citations. Answers without citations
    indicate potential hallucination and should be rejected.
    """
    answer: str = Field(
        description="The answer to the user's question, with inline [Source N] citations"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of source documents that support the answer"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model's confidence in the answer (0=low, 1=high)"
    )
    is_grounded: bool = Field(
        description="Whether the answer is fully grounded in retrieved documents"
    )
    refused: bool = Field(
        default=False,
        description="True if the system could not find sufficient context to answer"
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Explanation of why the question could not be answered"
    )


class QueryAnalysis(BaseModel):
    """Output from the query rewriting agent."""
    original_query: str
    rewritten_queries: list[str] = Field(
        description="Multiple reformulations of the query for better retrieval"
    )
    hypothetical_answer: str = Field(
        description="HyDE: a hypothetical answer used for embedding-based retrieval"
    )
    intent: str = Field(
        description="Detected intent: factual | comparison | summary | explanation"
    )
    needs_retrieval: bool = Field(
        description="False if this is a greeting/meta-question that doesn't need KB lookup"
    )


class ValidationResult(BaseModel):
    """Output from the validation agent."""
    is_sufficient: bool = Field(
        description="Whether retrieved context is sufficient to answer the query"
    )
    faithfulness_score: float = Field(
        ge=0.0, le=1.0,
        description="Estimated faithfulness: how well docs support the answer"
    )
    reasoning: str = Field(
        description="Brief explanation of the validation decision"
    )
