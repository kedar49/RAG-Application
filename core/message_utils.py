"""
Helpers for dealing with LangChain message content types.

`AIMessage.content` can be a string or a structured list depending on model/tooling
configuration, so these helpers normalize it to plain text for parsing and UI use.
"""

from typing import Any


def content_to_text(content: str | list[str | dict[str, Any]]) -> str:
    """Normalizes LangChain message content into a single plain string."""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue

        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue

        parts.append(str(item))

    return "\n".join(part for part in parts if part).strip()
