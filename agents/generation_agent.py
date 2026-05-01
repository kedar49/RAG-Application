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
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import RAGState
from config import settings

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
            # Fallback: generate citations from retrieved docs
            for num, meta in doc_meta.items():
                citations.append({
                    "source_number": num,
                    "source": meta["source"],
                    "page": meta.get("page"),
                    "excerpt": "",
                })

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

    logger.info(f"[GenerationAgent] Generating answer for: {query!r}")

    llm = ChatOllama(
        model=settings.generation_model,   # qwen2.5:7b
        base_url=settings.ollama_base_url,
        temperature=settings.generation_temperature,   # 0.1 — factual, not creative
    )

    messages = [
        SystemMessage(content=GENERATION_SYSTEM),
        HumanMessage(content=(
            f"Source Documents:\n{context}\n\n"
            f"Question: {query}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        answer, citations, confidence = _parse_generation_response(
            response.content, retrieved_docs
        )

        logger.info(
            f"[GenerationAgent] Generated answer ({len(answer)} chars), "
            f"citations={len(citations)}, confidence={confidence:.2f}"
        )

        return {
            **state,
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "refused": False,
            "refusal_reason": None,
        }

    except Exception as e:
        logger.error(f"[GenerationAgent] Error: {e}")
        return {
            **state,
            "answer": "",
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
