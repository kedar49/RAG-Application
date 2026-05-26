"""
Meta agent for non-retrieval conversations.

Handles greetings and product-help questions without pretending to be
document-grounded retrieval output.
"""

import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import RAGState
from config import settings
from core.message_utils import content_to_text

logger = logging.getLogger(__name__)

META_AGENT_SYSTEM = """You are GroundedRAG, a helpful assistant for a local RAG app.

You are being used only for messages that do NOT require knowledge-base retrieval.

Rules:
- Answer greetings naturally.
- Answer brief product-help questions about how this app works.
- Do not claim to have searched documents or found sources.
- If the user asks for document facts, tell them to ask a knowledge-base question instead.
- Keep responses concise and practical.
"""


def meta_agent_node(state: RAGState) -> RAGState:
    """Handles non-RAG conversational turns."""
    query = state.get("query", "")
    logger.info(f"[MetaAgent] Handling non-retrieval query: {query!r}")

    llm = ChatOllama(
        model=settings.validation_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )

    messages = [
        SystemMessage(content=META_AGENT_SYSTEM),
        HumanMessage(content=query),
    ]

    try:
        response = llm.invoke(messages)
        return {
            **state,
            "answer": content_to_text(response.content),
            "citations": [],
            "confidence": 0.9,
            "refused": False,
            "refusal_reason": None,
        }
    except Exception as e:
        logger.error(f"[MetaAgent] Error: {e}")
        return {
            **state,
            "answer": "I can help with the app and your uploaded documents, but I hit an internal error on this message.",
            "citations": [],
            "confidence": 0.0,
            "refused": True,
            "refusal_reason": f"Meta agent error: {e}",
            "error": str(e),
        }
