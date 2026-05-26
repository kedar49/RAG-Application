"""
Token-aware semantic chunking pipeline.

Uses RecursiveCharacterTextSplitter with a token-counting length function so
chunk boundaries better match model context budgets. It also preserves light
document structure such as headings and section paths in chunk metadata.
"""

import hashlib
from datetime import datetime

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

_ENCODING = None


def _get_encoding():
    """Lazily loads the tokenizer used for chunk sizing."""
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.get_encoding(settings.chunk_token_encoding)
        except KeyError:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Returns token count for chunk sizing and metadata."""
    return len(_get_encoding().encode(text, disallowed_special=()))


def _build_chunk_text(doc: Document) -> str:
    """
    Prefixes content with structural hints so retrieval can latch onto section names.
    """
    section_path = doc.metadata.get("section_path") or []
    section_heading = doc.metadata.get("section_heading")
    source_title = doc.metadata.get("source_title")

    prefix_parts: list[str] = []
    if source_title:
        prefix_parts.append(f"Title: {source_title}")
    if section_path:
        prefix_parts.append(f"Section: {' > '.join(section_path)}")
    elif section_heading:
        prefix_parts.append(f"Section: {section_heading}")

    if not prefix_parts:
        return doc.page_content.strip()

    prefix = "\n".join(prefix_parts).strip()
    body = doc.page_content.strip()
    return f"{prefix}\n\n{body}" if body else prefix


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a token-aware RecursiveCharacterTextSplitter that respects natural boundaries.
    Separator priority: explicit section markers > paragraphs > sentences > words.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=count_tokens,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",
        ],
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    source_name: str,
    source_type: str = "pdf",
) -> list[Document]:
    """
    Splits a list of documents into token-aware chunks and enriches metadata.
    """
    splitter = get_text_splitter()
    chunks: list[Document] = []
    global_chunk_index = 0

    for doc_index, doc in enumerate(documents):
        page = doc.metadata.get("page", doc.metadata.get("page_number", None))
        page_marker = page if page is not None else doc_index
        structured_doc = Document(
            page_content=_build_chunk_text(doc),
            metadata=dict(doc.metadata),
        )

        splits = splitter.split_documents([structured_doc])
        for chunk_index_within_doc, chunk in enumerate(splits):
            content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:12]
            chunk_id = (
                f"{source_name}_p{page_marker}_d{doc_index}_"
                f"c{chunk_index_within_doc}_{content_hash}"
            )

            chunk.metadata.update({
                "source": source_name,
                "source_type": source_type,
                "chunk_index": global_chunk_index,
                "chunk_index_within_document": chunk_index_within_doc,
                "source_document_index": doc_index,
                "chunk_id": chunk_id,
                "ingested_at": datetime.utcnow().isoformat(),
                "page": page,
                "token_count": count_tokens(chunk.page_content),
            })
            chunks.append(chunk)
            global_chunk_index += 1

    return chunks
