"""
Semantic chunking pipeline.

Uses LangChain's RecursiveCharacterTextSplitter with carefully tuned
parameters rather than dumb fixed-size chunking. This preserves sentence
and paragraph boundaries, which is critical for accurate retrieval.

Why semantic > fixed chunking:
- Fixed chunks split mid-sentence → broken context → worse embeddings
- Semantic chunks preserve logical units → better embeddings → better retrieval
"""

import hashlib
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a RecursiveCharacterTextSplitter that respects natural boundaries.
    Separator priority: paragraphs > sentences > words > characters.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,         # ~512 tokens per chunk
        chunk_overlap=settings.chunk_overlap,   # 50-token overlap for continuity
        length_function=len,
        separators=[
            "\n\n",     # paragraph breaks (highest priority)
            "\n",       # line breaks
            ". ",       # sentence ends
            "! ",
            "? ",
            "; ",
            ", ",
            " ",        # word boundaries
            "",         # character level (last resort)
        ],
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    source_name: str,
    source_type: str = "pdf",
) -> list[Document]:
    """
    Splits a list of documents into smaller, well-bounded chunks.
    Enriches each chunk with metadata for citation and filtering.

    Args:
        documents:   List of Document objects from a loader.
        source_name: Display name for the source (filename or URL).
        source_type: "pdf" | "web" | "text"

    Returns:
        List of chunked Documents with enriched metadata.
    """
    splitter = get_text_splitter()
    chunks: list[Document] = []

    for doc in documents:
        splits = splitter.split_documents([doc])
        for i, chunk in enumerate(splits):
            # Build a stable chunk ID for deduplication in the vector store
            content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:12]
            chunk_id = f"{source_name}_{i}_{content_hash}"

            chunk.metadata.update({
                "source": source_name,
                "source_type": source_type,
                "chunk_index": i,
                "chunk_id": chunk_id,
                "ingested_at": datetime.utcnow().isoformat(),
                # Preserve page number if available from loader
                "page": doc.metadata.get("page", doc.metadata.get("page_number", None)),
            })
            chunks.append(chunk)

    return chunks
