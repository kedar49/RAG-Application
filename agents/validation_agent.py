"""
Validation Agent — the anti-hallucination gate.

Before generating an answer, this agent checks:
  1. Does the retrieved context actually contain information relevant to the query?
  2. Is there enough context to answer confidently?

If the context is insufficient, the agent sets validation_passed=False
and the orchestrator routes to a refusal response instead of generation.

This is the MOST IMPORTANT node for preventing hallucinations.
Without this gate, the generation LLM will "make up" an answer
when the retrieved documents don't contain the answer.

Uses phi4-mini:3.8b — lightweight but strong reasoning model.
"""

import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import RAGState
from config import settings

logger = logging.getLogger(__name__)

VALIDATION_SYSTEM = """You are a strict validation agent for a RAG system. 
Your role is to determine if the provided context documents contain sufficient, relevant information to answer the user's question.

Rules:
- Be STRICT. If the documents are only tangentially related, mark as insufficient.
- "Sufficient" means the documents directly address the question with specific facts, data, or explanations.
- If even ONE document clearly answers the question, that may be sufficient.
- Never say "I think" — base your judgment purely on what's in the context.

You MUST respond in valid JSON only. No extra text.

JSON Template:
{
  "is_sufficient": true,
  "faithfulness_score": 0.85,
  "reasoning": "<brief explanation of your decision>"
}

faithfulness_score: 0.0 = no relevant info in context, 1.0 = context perfectly answers the question"""


def validation_agent_node(state: RAGState) -> RAGState:
    """
    LangGraph node: faithfulness validation gate.
    
    Reads:  state["query"], state["retrieval_context"]
    Writes: state["validation_passed"], state["faithfulness_score"],
            state["validation_reasoning"]
    """
    query = state.get("query", "")
    context = state.get("retrieval_context", "")

    # If no context was retrieved at all, fail immediately
    if not context.strip():
        logger.warning("[ValidationAgent] Empty context — failing validation.")
        return {
            **state,
            "validation_passed": False,
            "faithfulness_score": 0.0,
            "validation_reasoning": "No documents were retrieved from the knowledge base.",
        }

    llm = ChatOllama(
        model=settings.validation_model,   # phi4-mini:3.8b
        base_url=settings.ollama_base_url,
        temperature=0.0,                   # deterministic — validation should be consistent
        format="json",
    )

    # Truncate context for validation prompt (don't need full 8k for validation)
    context_snippet = context[:3000] if len(context) > 3000 else context

    messages = [
        SystemMessage(content=VALIDATION_SYSTEM),
        HumanMessage(content=(
            f"User Question: {query}\n\n"
            f"Retrieved Context:\n{context_snippet}\n\n"
            f"Is this context sufficient to answer the question?"
        )),
    ]

    try:
        response = llm.invoke(messages)
        parsed = json.loads(response.content)

        is_sufficient = parsed.get("is_sufficient", False)
        faithfulness_score = float(parsed.get("faithfulness_score", 0.0))
        reasoning = parsed.get("reasoning", "")

        # Apply threshold — even if LLM says True, check the score
        if faithfulness_score < settings.faithfulness_threshold:
            is_sufficient = False
            reasoning = f"Faithfulness score {faithfulness_score:.2f} below threshold {settings.faithfulness_threshold}. {reasoning}"

        logger.info(
            f"[ValidationAgent] sufficient={is_sufficient}, "
            f"score={faithfulness_score:.2f}, threshold={settings.faithfulness_threshold}"
        )

        return {
            **state,
            "validation_passed": is_sufficient,
            "faithfulness_score": faithfulness_score,
            "validation_reasoning": reasoning,
        }

    except (json.JSONDecodeError, ValueError) as e:
        # On parse failure, be CONSERVATIVE — fail validation
        logger.warning(f"[ValidationAgent] Parse failed ({e}), defaulting to fail.")
        return {
            **state,
            "validation_passed": False,
            "faithfulness_score": 0.0,
            "validation_reasoning": f"Validation failed to parse response: {e}",
        }
