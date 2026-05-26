"""
PDF ingestion pipeline.

Full pipeline: PDF file → text extraction → semantic chunking →
embeddings → pgvector storage.

Uses pypdf for extraction and the core chunker for semantic splitting.
"""

import logging
import tempfile
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pypdf import PdfReader

from core.chunker import chunk_documents
from core.vector_store import add_documents, source_exists

logger = logging.getLogger(__name__)


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    """
    Ingests a PDF file into the vector store.
    
    Args:
        file_bytes: Raw bytes of the PDF file (from Streamlit uploader).
        filename:   Original filename for citation metadata.
    
    Returns:
        Number of chunks successfully stored.
    """
    source_name = Path(filename).stem  # e.g., "my_document" from "my_document.pdf"
    if source_exists(source_name):
        logger.info(f"Skipping duplicate PDF source: {source_name}")
        return 0

    # Write to temp file — PyPDFLoader needs a file path
    with tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False, mode="wb"
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        pdf_metadata = reader.metadata or {}
        source_title = pdf_metadata.get("/Title") or source_name
        total_pages = len(reader.pages)

        # Load PDF pages
        loader = PyPDFLoader(tmp_path)
        raw_docs: list[Document] = loader.load()

        if not raw_docs:
            logger.warning(f"No text extracted from {filename}")
            return 0

        logger.info(f"Extracted {len(raw_docs)} pages from {filename}")

        # Tag each page with the display source name
        for index, doc in enumerate(raw_docs):
            doc.metadata["source"] = source_name
            doc.metadata["filename"] = filename
            doc.metadata["source_title"] = source_title
            doc.metadata["total_pages"] = total_pages
            doc.metadata["page_number"] = doc.metadata.get("page", index)
            if pdf_metadata.get("/Author"):
                doc.metadata["author"] = pdf_metadata.get("/Author")

        # Semantic chunking
        chunks = chunk_documents(raw_docs, source_name=source_name, source_type="pdf")
        logger.info(f"Chunked into {len(chunks)} chunks")

        # Embed + store
        add_documents(chunks)
        logger.info(f"Stored {len(chunks)} chunks from {filename}")
        return len(chunks)

    except Exception as e:
        logger.error(f"PDF ingestion failed for {filename}: {e}")
        raise

    finally:
        os.unlink(tmp_path)


def ingest_pdf_from_path(file_path: str) -> int:
    """
    Convenience: ingest a PDF directly from a filesystem path.
    Useful for batch ingestion scripts.
    """
    path = Path(file_path)
    with open(path, "rb") as f:
        return ingest_pdf(f.read(), path.name)
