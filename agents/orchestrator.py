"""
Orchestrator — the LangGraph graph definition.

Wires together all 5 agents into a directed conditional graph:

  ┌──────────────────────────────────────────────────────────┐
  │                      START                               │
  │                        │                                 │
  │                  [query_agent]                           │
  │                        │                                 │
  │              needs_retrieval?                            │
  │               Yes │     │ No                             │
  │        [retrieval_agent] │                               │
  │                   │      │                               │
  │          [validation_agent] │                            │
  │                   │        │                             │
  │         validation_passed? │                             │
  │            Yes │   │ No    │                             │
  │  [generation_agent] [refusal_node]                       │
  │                │        │   │                            │
  │                └────────┘   │                            │
  │                     └───────┘                            │
  │                          END                             │
  └──────────────────────────────────────────────────────────┘

The graph is compiled once at import time and reused per session.
"""

import logging
from typing import cast
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agents.state import RAGState
from agents.meta_agent import meta_agent_node
from agents.query_agent import query_agent_node
from agents.retrieval_agent import retrieval_agent_node
from agents.validation_agent import validation_agent_node
from agents.generation_agent import generation_agent_node, refusal_node

logger = logging.getLogger(__name__)

_RAG_GRAPH: CompiledStateGraph | None = None


# ── Conditional edge functions ────────────────────────────────────────────────

def route_after_query(state: RAGState) -> str:
    """Route: does this query need retrieval from the knowledge base?"""
    if not state.get("needs_retrieval", True):
        # Greeting or meta-question — go directly to generation (no retrieval)
        logger.debug("[Orchestrator] Skipping retrieval (query doesn't need KB)")
        return "meta"
    return "retrieve"


def route_after_validation(state: RAGState) -> str:
    """Route: did the retrieved context pass the faithfulness check?"""
    if state.get("validation_passed", False):
        logger.debug("[Orchestrator] Validation passed → generating answer")
        return "generate"
    logger.debug("[Orchestrator] Validation failed → refusing")
    return "refuse"


# ── Graph construction ────────────────────────────────────────────────────────

def build_rag_graph() -> CompiledStateGraph:
    """
    Builds and returns the compiled LangGraph RAG pipeline.
    Call this once at startup and reuse the returned graph.
    """
    graph = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("query_agent", query_agent_node)
    graph.add_node("meta_agent", meta_agent_node)
    graph.add_node("retrieval_agent", retrieval_agent_node)
    graph.add_node("validation_agent", validation_agent_node)
    graph.add_node("generation_agent", generation_agent_node)
    graph.add_node("refusal_node", refusal_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("query_agent")

    # ── Conditional: after query analysis ─────────────────────────────────────
    graph.add_conditional_edges(
        "query_agent",
        route_after_query,
        {
            "retrieve": "retrieval_agent",
            "meta": "meta_agent",
        },
    )

    # ── Linear: retrieval → validation ────────────────────────────────────────
    graph.add_edge("retrieval_agent", "validation_agent")

    # ── Conditional: after validation ─────────────────────────────────────────
    graph.add_conditional_edges(
        "validation_agent",
        route_after_validation,
        {
            "generate": "generation_agent",
            "refuse": "refusal_node",
        },
    )

    # ── Terminal edges ────────────────────────────────────────────────────────
    graph.add_edge("meta_agent", END)
    graph.add_edge("generation_agent", END)
    graph.add_edge("refusal_node", END)

    return graph.compile()


def run_rag_pipeline(query: str, chat_history: list | None = None) -> RAGState:
    """
    Main entry point: run the full multi-agent RAG pipeline for a query.
    
    Args:
        query:        The user's question.
        chat_history: Optional list of previous messages for context.
    
    Returns:
        Final RAGState with answer, citations, confidence, etc.
    """
    global _RAG_GRAPH
    if _RAG_GRAPH is None:
        _RAG_GRAPH = build_rag_graph()

    initial_state: RAGState = {
        "query": query,
        "messages": chat_history or [],
        "rewritten_queries": [],
        "hypothetical_answer": "",
        "query_intent": "factual",
        "needs_retrieval": True,
        "retrieved_docs": [],
        "retrieval_context": "",
        "retrieval_debug": [],
        "validation_passed": False,
        "faithfulness_score": 0.0,
        "validation_reasoning": "",
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "refused": False,
        "refusal_reason": None,
        "error": None,
    }

    logger.info(f"[Orchestrator] Starting RAG pipeline for query: {query!r}")

    try:
        final_state = _RAG_GRAPH.invoke(initial_state)
        logger.info(
            f"[Orchestrator] Pipeline complete. "
            f"refused={final_state.get('refused')}, "
            f"confidence={final_state.get('confidence', 0):.2f}"
        )
        return cast(RAGState, final_state)
    except Exception as e:
        logger.error(f"[Orchestrator] Pipeline error: {e}")
        return cast(RAGState, {
            **initial_state,
            "answer": f"An error occurred: {e}",
            "refused": True,
            "refusal_reason": str(e),
            "error": str(e),
        })
