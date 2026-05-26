"""
Query Rewriting Agent.

Transforms the raw user query into multiple better-formed queries
using three techniques:
  1. HyDE (Hypothetical Document Embeddings) — generates a hypothetical answer
     and embeds it to find semantically similar docs (bridges vocabulary gaps).
  2. Multi-query expansion — 3 reformulations of the query for broader coverage.
  3. Intent classification — routes to appropriate retrieval strategy.

Why this matters:
  User query: "what does the report say about Q3?"
  → Often too vague for direct vector search
  → HyDE generates a plausible Q3 summary → better semantic alignment
  → Multi-query catches edge cases the original query would miss
"""

import json
import logging
from pydantic import ValidationError
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import RAGState
from config import settings
from core.message_utils import content_to_text
from schemas import QueryAnalysis

logger = logging.getLogger(__name__)

# System prompt for the query rewriting agent
QUERY_AGENT_SYSTEM = """You are a query analysis specialist for a RAG system.
Your job is to analyze a user query and produce:
1. A hypothetical answer (HyDE technique) — write what a perfect answer would look like, even if you're not sure. This is used to find relevant documents.
2. Three reformulated versions of the query — rephrase it differently to improve retrieval coverage.
3. The intent type: factual | comparison | summary | explanation | other
4. Whether the query needs document retrieval (False only for greetings, system questions, etc.)

You MUST respond in valid JSON format exactly as shown in the template. No extra text.

JSON Template:
{
  "hypothetical_answer": "<write a 2-4 sentence plausible answer to the query>",
  "rewritten_queries": [
    "<rephrasing 1>",
    "<rephrasing 2>",
    "<rephrasing 3>"
  ],
  "intent": "<factual|comparison|summary|explanation|other>",
  "needs_retrieval": true
}"""


def _format_chat_history(messages: list | None) -> str:
    """Formats recent chat turns for history-aware query analysis."""
    if not messages:
        return "No prior conversation."

    formatted: list[str] = []
    for message in messages[-6:]:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
        else:
            role = getattr(message, "type", getattr(message, "role", "user"))
            content = getattr(message, "content", "")

        if content:
            formatted.append(f"{role.title()}: {content}")

    return "\n".join(formatted) if formatted else "No prior conversation."


def query_agent_node(state: RAGState) -> RAGState:
    """
    LangGraph node: query analysis and rewriting.
    
    Reads:  state["query"]
    Writes: state["rewritten_queries"], state["hypothetical_answer"],
            state["query_intent"], state["needs_retrieval"]
    """
    query = state.get("query", "")
    chat_history = state.get("messages", [])
    logger.info(f"[QueryAgent] Processing query: {query!r}")

    llm = ChatOllama(
        model=settings.validation_model,   # phi4-mini:3.8b — fast, good reasoning
        base_url=settings.ollama_base_url,
        temperature=0.3,                   # slight creativity for reformulations
        format="json",                     # enforce JSON output
    )

    messages = [
        SystemMessage(content=QUERY_AGENT_SYSTEM),
        HumanMessage(content=(
            f"Conversation so far:\n{_format_chat_history(chat_history)}\n\n"
            f"Latest user query: {query}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        response_text = content_to_text(response.content)
        parsed = QueryAnalysis.model_validate({
            "original_query": query,
            **json.loads(response_text),
        })

        rewritten = parsed.rewritten_queries or [query]
        hypothetical = parsed.hypothetical_answer or query
        intent = parsed.intent or "factual"
        needs_retrieval = parsed.needs_retrieval

        logger.info(
            f"[QueryAgent] intent={intent}, needs_retrieval={needs_retrieval}, "
            f"reformulations={len(rewritten)}"
        )

        return {
            **state,
            "rewritten_queries": rewritten,
            "hypothetical_answer": hypothetical,
            "query_intent": intent,
            "needs_retrieval": needs_retrieval,
        }

    except (json.JSONDecodeError, KeyError, ValidationError) as e:
        # Graceful fallback: use original query if parsing fails
        logger.warning(f"[QueryAgent] JSON parse failed ({e}), using original query")
        return {
            **state,
            "rewritten_queries": [query],
            "hypothetical_answer": query,
            "query_intent": "factual",
            "needs_retrieval": True,
        }
