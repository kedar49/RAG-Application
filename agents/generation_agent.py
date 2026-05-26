"""
Generation Agent.

Produces the final answer with:
  - Inline [Source N] citations mapping claims to retrieved documents
  - Chain-of-thought reasoning for complex questions
  - Low temperature (0.1) for factual, deterministic answers
  - Structured output extracting citations for UI rendering

Uses qwen2.5:7b — best instruction-following quality at the 7B class.
Temperature is deliberately set low (0.1) — RAG is a factual task,
not a creative one. Higher temperature = more hallucination risk.
"""

import json
import logging
import re
from pydantic import ValidationError
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import RAGState
from config import settings
from schemas import RAGResponse

logger = logging.getLogger(__name__)

GENERATION_SYSTEM = """You are RAGit, a precise document-grounded AI assistant.

Your ONLY job is to answer questions based on the provided source documents.

STRICT RULES:
1. Only use information from the provided [Source N] documents.
2. Every factual claim MUST reference a source using [Source N] notation inline.
3. If multiple sources support a claim, cite all of them: [Source 1][Source 3].
4. NEVER add information from your training data. Only what's in the sources.
5. If the sources don't fully answer the question, say what IS in the sources, cite it, and note what's missing.
6. Be concise and direct. No filler phrases like "Based on the documents..." or "According to the context..."
7. Format your answer in clean Markdown.

After your answer, output a JSON block in this exact format (no other text after it):
```json
{
  "citations": [
    {"source_number": 1, "source_name": "<filename/url from the source>", "excerpt": "<key quote from the source>"},
    ...
  ],
  "confidence": 0.9
}
```"""


def _format_chat_history(messages: list | None) -> str:
    """Formats recent chat turns so follow-up answers keep conversation context."""
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


def _extract_citations_from_docs(retrieved_docs: list) -> dict[int, dict]:
    """Maps source number (1-indexed) to document metadata."""
    citation_map: dict[int, dict] = {}
    for i, doc in enumerate(retrieved_docs, start=1):
        citation_map[i] = {
            "source": doc.metadata.get("source", f"Source {i}"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
        }
    return citation_map


def _extract_inline_citation_numbers(answer: str) -> set[int]:
    """Collects all [Source N] citations used inline in the answer body."""
    return {
        int(match)
        for match in re.findall(r"\[Source\s+(\d+)\]", answer)
    }


def _is_answer_grounded(answer: str, citations: list[dict], doc_meta: dict[int, dict]) -> bool:
    """
    Conservative grounding check:
    - answer must cite at least one valid source inline
    - every structured citation must refer to a retrieved source
    """
    inline_sources = _extract_inline_citation_numbers(answer)
    if not inline_sources:
        return False

    if not inline_sources.issubset(set(doc_meta)):
        return False

    citation_sources = {citation["source_number"] for citation in citations}
    if not citation_sources:
        return False

    if not inline_sources.issubset(citation_sources):
        return False

    for citation in citations:
        if citation["source_number"] not in doc_meta:
            return False

    return True


def _grounding_refusal_message(query: str, reason: str) -> str:
    """Builds a user-facing refusal when generation is not grounded enough."""
    return (
        "I couldn't produce a fully grounded answer from the retrieved documents.\n\n"
        f"**Question:** {query}\n\n"
        f"**Why I stopped:** {reason}\n\n"
        "Try rephrasing the question or adding documents that address it more directly."
    )


def _parse_generation_response(
    content: str,
    retrieved_docs: list,
) -> tuple[str, list[dict], float]:
    """
    Parses the LLM output, splitting the answer from the JSON citation block.
    
    Returns:
        (answer_text, citations_list, confidence_score)
    """
    # Try to extract JSON block from the response
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, content)

    answer = content
    citations: list[dict] = []
    confidence = 0.8

    doc_meta = _extract_citations_from_docs(retrieved_docs)

    if match:
        json_str = match.group(1)
        answer = content[: match.start()].strip()
        try:
            parsed = json.loads(json_str)
            raw_citations = parsed.get("citations", [])
            confidence = float(parsed.get("confidence", 0.8))

            for c in raw_citations:
                num = c.get("source_number", 0)
                meta = doc_meta.get(num, {})
                citations.append({
                    "source_number": num,
                    "source": meta.get("source", c.get("source_name", "Unknown")),
                    "page": meta.get("page"),
                    "excerpt": c.get("excerpt", ""),
                })
        except (json.JSONDecodeError, ValueError):
            citations = []

    return answer, citations, confidence


def generation_agent_node(state: RAGState) -> RAGState:
    """
    LangGraph node: final answer generation with citations.
    
    Reads:  state["query"], state["retrieval_context"], state["retrieved_docs"]
    Writes: state["answer"], state["citations"], state["confidence"]
    """
    query = state.get("query", "")
    context = state.get("retrieval_context", "")
    retrieved_docs = state.get("retrieved_docs", [])
    chat_history = state.get("messages", [])

    logger.info(f"[GenerationAgent] Generating answer for: {query!r}")

    llm = ChatOllama(
        model=settings.generation_model,   # qwen2.5:7b
        base_url=settings.ollama_base_url,
        temperature=settings.generation_temperature,   # 0.1 — factual, not creative
    )

    messages = [
        SystemMessage(content=GENERATION_SYSTEM),
        HumanMessage(content=(
            f"Conversation so far:\n{_format_chat_history(chat_history)}\n\n"
            f"Source Documents:\n{context}\n\n"
            f"Question: {query}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        answer, citations, confidence = _parse_generation_response(
            response.content, retrieved_docs
        )
        doc_meta = _extract_citations_from_docs(retrieved_docs)
        is_grounded = _is_answer_grounded(answer, citations, doc_meta)
        validated = RAGResponse.model_validate({
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "is_grounded": is_grounded,
            "refused": not is_grounded,
            "refusal_reason": None if is_grounded else "Generated answer was not fully grounded in retrieved sources.",
        })

        if not validated.is_grounded:
            logger.warning("[GenerationAgent] Rejecting ungrounded answer.")
            return {
                **state,
                "answer": _grounding_refusal_message(query, validated.refusal_reason or "Missing valid source support."),
                "citations": [],
                "confidence": 0.0,
                "refused": True,
                "refusal_reason": validated.refusal_reason,
                "error": validated.refusal_reason,
            }

        logger.info(
            f"[GenerationAgent] Generated answer ({len(validated.answer)} chars), "
            f"citations={len(validated.citations)}, confidence={validated.confidence:.2f}"
        )

        return {
            **state,
            "answer": validated.answer,
            "citations": [citation.model_dump() for citation in validated.citations],
            "confidence": validated.confidence,
            "refused": False,
            "refusal_reason": None,
        }

    except (ValidationError, Exception) as e:
        logger.error(f"[GenerationAgent] Error: {e}")
        return {
            **state,
            "answer": _grounding_refusal_message(query, f"Generation error: {e}"),
            "citations": [],
            "confidence": 0.0,
            "refused": True,
            "refusal_reason": f"Generation error: {e}",
            "error": str(e),
        }


def refusal_node(state: RAGState) -> RAGState:
    """
    LangGraph node: polite refusal when context is insufficient.
    
    This runs instead of generation_agent_node when validation fails.
    It explains WHY the system can't answer instead of hallucinating.
    """
    query = state.get("query", "")
    faithfulness = state.get("faithfulness_score", 0.0)
    validation_reasoning = state.get("validation_reasoning", "")

    logger.info(
        f"[RefusalNode] Refusing to answer. Score={faithfulness:.2f}. "
        f"Reason: {validation_reasoning}"
    )

    refusal_message = (
        "I don't have sufficient information in my knowledge base to answer this question confidently.\n\n"
        f"**What I searched for:** {query}\n\n"
        "**What happened:** The documents I retrieved don't directly address this question. "
        "This could mean:\n"
        "- The answer isn't in the uploaded documents\n"
        "- The question needs different/additional documents\n"
        "- Try rephrasing your question\n\n"
        "*I won't guess or make up an answer — only grounded responses are shown.*"
    )

    return {
        **state,
        "answer": refusal_message,
        "citations": [],
        "confidence": 0.0,
        "refused": True,
        "refusal_reason": validation_reasoning,
    }
