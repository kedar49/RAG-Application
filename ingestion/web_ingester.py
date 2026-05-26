"""
Web / URL ingestion pipeline.

Scrapes a URL, extracts section-aware text, and stores chunked documents with
structural metadata such as page title and heading path.
"""

import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document

from core.chunker import chunk_documents
from core.vector_store import add_documents, source_exists

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAGit/1.0; +https://github.com/ragit)"
    )
}
REQUEST_TIMEOUT = 10
HEADING_TAGS = {"h1", "h2", "h3"}
CONTENT_TAGS = {"p", "li", "pre", "blockquote"}


def _canonical_source_name(url: str) -> str:
    """Creates a stable display/source key from the URL."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.netloc}{path}"


def _clean_soup(html: str) -> BeautifulSoup:
    """Removes noisy HTML before content extraction."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
        tag.decompose()
    return soup


def _heading_level(tag_name: str) -> int:
    """Turns h1/h2/h3 into comparable depth values."""
    return int(tag_name[1]) if len(tag_name) == 2 and tag_name[1].isdigit() else 99


def _extract_section_documents(html: str, url: str) -> list[Document]:
    """
    Extracts section-aware documents from HTML.

    Each section groups nearby paragraph/list/preformatted content under the most
    recent heading path.
    """
    soup = _clean_soup(html)
    root = soup.find("main") or soup.find("article") or soup.find("body") or soup
    source_title = (soup.title.string or "").strip() if soup.title and soup.title.string else None

    documents: list[Document] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_heading = source_title or "Page"

    def flush_current_section():
        nonlocal current_lines
        content = "\n".join(line for line in current_lines if line).strip()
        if not content:
            current_lines = []
            return

        section_path = heading_stack[:] if heading_stack else [current_heading]
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": _canonical_source_name(url),
                    "url": url,
                    "source_type": "web",
                    "source_title": source_title or _canonical_source_name(url),
                    "section_heading": current_heading,
                    "section_path": section_path,
                },
            )
        )
        current_lines = []

    for node in root.descendants:
        if not isinstance(node, Tag):
            continue

        tag_name = node.name.lower()
        if tag_name in HEADING_TAGS:
            flush_current_section()
            text = node.get_text(" ", strip=True)
            if not text:
                continue

            level = _heading_level(tag_name)
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(text)
            current_heading = text
            continue

        if tag_name in CONTENT_TAGS:
            text = node.get_text(" ", strip=True)
            if text:
                current_lines.append(text)

    flush_current_section()

    if documents:
        return documents

    fallback_text = root.get_text(separator="\n", strip=True)
    if not fallback_text:
        return []

    return [
        Document(
            page_content=fallback_text,
            metadata={
                "source": _canonical_source_name(url),
                "url": url,
                "source_type": "web",
                "source_title": source_title or _canonical_source_name(url),
            },
        )
    ]


def ingest_url(url: str) -> int:
    """
    Scrapes and ingests a single URL into the vector store.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")

    source_name = _canonical_source_name(url)
    if source_exists(source_name):
        logger.info(f"Skipping duplicate URL source: {source_name}")
        return 0

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

    if "text/plain" in content_type:
        raw_docs = [
            Document(
                page_content=response.text.strip(),
                metadata={
                    "source": source_name,
                    "url": url,
                    "source_type": "web",
                    "source_title": source_name,
                },
            )
        ]
    else:
        raw_docs = _extract_section_documents(response.text, url)

    if not raw_docs:
        raise ValueError(f"No readable text extracted from {url}")

    total_chars = sum(len(doc.page_content) for doc in raw_docs)
    logger.info(f"Extracted {total_chars} characters from {url} across {len(raw_docs)} sections")

    chunks = chunk_documents(raw_docs, source_name=source_name, source_type="web")
    logger.info(f"Chunked {url} into {len(chunks)} chunks")

    add_documents(chunks)
    logger.info(f"Stored {len(chunks)} chunks from {url}")
    return len(chunks)
