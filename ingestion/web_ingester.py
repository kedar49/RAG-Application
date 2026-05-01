"""
Web / URL ingestion pipeline.

Scrapes a URL using BeautifulSoup, cleans the content,
chunks it, and stores it in the vector store with the URL as source.
"""

import logging
import requests
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from core.chunker import chunk_documents
from core.vector_store import add_documents

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAGit/1.0; +https://github.com/ragit)"
    )
}
REQUEST_TIMEOUT = 10  # seconds


def _extract_text_from_html(html: str, url: str) -> str:
    """
    Extracts clean readable text from HTML.
    Removes scripts, styles, nav, footer, and other non-content elements.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "iframe", "noscript"]):
        tag.decompose()

    # Try to find the main content area first
    main = soup.find("main") or soup.find("article") or soup.find("body")
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(
        separator="\n", strip=True
    )

    # Collapse excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def ingest_url(url: str) -> int:
    """
    Scrapes and ingests a single URL into the vector store.
    
    Args:
        url: The URL to scrape and ingest.
    
    Returns:
        Number of chunks successfully stored.
    
    Raises:
        ValueError: If the URL is invalid or unreachable.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")

    source_name = parsed.netloc + parsed.path

    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch {url}: {e}") from e

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ValueError(
            f"Unsupported content type: {content_type}. Only HTML/text is supported."
        )

    text = _extract_text_from_html(response.text, url)
    if not text.strip():
        raise ValueError(f"No readable text extracted from {url}")

    logger.info(f"Extracted {len(text)} characters from {url}")

    # Wrap in a Document
    raw_doc = Document(
        page_content=text,
        metadata={
            "source": source_name,
            "url": url,
            "source_type": "web",
        },
    )

    # Chunk and store
    chunks = chunk_documents([raw_doc], source_name=source_name, source_type="web")
    logger.info(f"Chunked {url} into {len(chunks)} chunks")

    add_documents(chunks)
    logger.info(f"Stored {len(chunks)} chunks from {url}")
    return len(chunks)
