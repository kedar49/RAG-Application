"""
RAGit — Multi-Agent RAG System
Upgraded Streamlit UI with:
  - Real-time pipeline status indicators
  - Grounded citations panel
  - Confidence scores
  - Faithfulness indicator
  - Dark-mode friendly styling
  - Knowledge base management
"""

import logging
import streamlit as st

from agents.orchestrator import run_rag_pipeline
from core.vector_store import clear_collection, list_sources
from ingestion.pdf_ingester import ingest_pdf
from ingestion.web_ingester import ingest_url
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAGit — Multi-Agent RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a4a;
}
section[data-testid="stSidebar"] * {
    color: #e0e0f5 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #a0a0c5 !important;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Main chat area ─────────────────────────────────────────── */
.main .block-container {
    padding-top: 1rem;
    max-width: 900px;
}

/* ── Chat messages ───────────────────────────────────────────── */
.stChatMessage {
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
}

/* ── Citations card ─────────────────────────────────────────── */
.citation-card {
    background: linear-gradient(135deg, #1e1e3a 0%, #16213e 100%);
    border: 1px solid #2a2a5a;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
}
.citation-card .source-label {
    color: #7c83ff;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.citation-card .excerpt {
    color: #b0b0d0;
    font-style: italic;
    margin-top: 0.25rem;
    line-height: 1.5;
}

/* ── Confidence badge ───────────────────────────────────────── */
.confidence-high  { color: #4ade80; font-weight: 600; }
.confidence-mid   { color: #fbbf24; font-weight: 600; }
.confidence-low   { color: #f87171; font-weight: 600; }

/* ── Refusal banner ─────────────────────────────────────────── */
.refusal-banner {
    background: linear-gradient(135deg, #2d1b1b, #1f1515);
    border: 1px solid #5a2020;
    border-radius: 10px;
    padding: 1rem;
    color: #ff9999;
}

/* ── Pipeline status ────────────────────────────────────────── */
.pipeline-step {
    display: inline-block;
    background: #1e1e3a;
    border: 1px solid #3a3a6a;
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    color: #9090cc;
    margin-right: 0.3rem;
}

/* ── Model badge ────────────────────────────────────────────── */
.model-badge {
    background: linear-gradient(135deg, #2d1f6e, #1a1040);
    border: 1px solid #4a3a9a;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    color: #a090ff;
    font-weight: 500;
}

/* ── Section headers ─────────────────────────────────────────── */
.section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6060aa;
    font-weight: 600;
    margin: 1rem 0 0.4rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid #2a2a4a;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "messages": [],
        "url_key": 0,
        "file_key": 100,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Confidence badge renderer ──────────────────────────────────────────────────
def _render_confidence(confidence: float, refused: bool):
    if refused:
        st.markdown(
            '<span class="confidence-low">⚠ Insufficient context</span>',
            unsafe_allow_html=True,
        )
        return
    if confidence >= 0.75:
        cls, label = "confidence-high", "High"
    elif confidence >= 0.5:
        cls, label = "confidence-mid", "Medium"
    else:
        cls, label = "confidence-low", "Low"

    st.markdown(
        f'<span class="{cls}">● {label} confidence ({confidence:.0%})</span>',
        unsafe_allow_html=True,
    )


# ── Citations renderer ─────────────────────────────────────────────────────────
def _render_citations(citations: list[dict]):
    if not citations:
        return
    st.markdown(
        '<div class="section-header">📎 Sources</div>', unsafe_allow_html=True
    )
    for c in citations:
        num = c.get("source_number", "?")
        source = c.get("source", "Unknown")
        page = c.get("page")
        excerpt = c.get("excerpt", "")

        page_str = f" · Page {page}" if page else ""
        excerpt_html = (
            f'<div class="excerpt">"{excerpt}"</div>' if excerpt else ""
        )
        st.markdown(
            f"""
            <div class="citation-card">
                <span class="source-label">[Source {num}] {source}{page_str}</span>
                {excerpt_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Sidebar ────────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.markdown(
            "## 🤖 RAGit\n*Multi-Agent RAG System*",
        )
        st.divider()

        # Model info
        st.markdown('<div class="section-header">Models</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="model-badge">🧠 {settings.generation_model}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="model-badge">🔍 {settings.validation_model}</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Embedder: {settings.embedding_model}")
        st.divider()

        # Knowledge base ingestion
        st.markdown('<div class="section-header">Knowledge Base</div>', unsafe_allow_html=True)

        # URL ingestion
        url_input = st.text_input(
            "Add URL",
            placeholder="https://...",
            key=f"url_{st.session_state['url_key']}",
        )
        if st.button("📥 Ingest URL", use_container_width=True):
            if url_input:
                with st.spinner("Scraping and embedding..."):
                    try:
                        n = ingest_url(url_input)
                        st.success(f"✅ Added {n} chunks from URL")
                        st.session_state["url_key"] += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")

        # PDF ingestion
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key=f"file_{st.session_state['file_key']}",
        )
        if uploaded_file:
            file_key = f"ingested_{uploaded_file.name}"
            if file_key not in st.session_state:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        n = ingest_pdf(uploaded_file.read(), uploaded_file.name)
                        st.success(f"✅ Added {n} chunks from PDF")
                        st.session_state[file_key] = True
                    except Exception as e:
                        st.error(f"❌ {e}")

        st.divider()

        # Clear KB
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True, type="secondary"):
            try:
                clear_collection()
                st.session_state.clear()
                _init_session()
                st.success("Knowledge base cleared")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

        # Clear chat
        if st.button("💬 Clear Chat", use_container_width=True, type="secondary"):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        st.markdown(
            '<div class="section-header">Pipeline Config</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Retrieval top-K: **{settings.retrieval_top_k}** → "
            f"Rerank to: **{settings.rerank_top_k}** → "
            f"Faithfulness threshold: **{settings.faithfulness_threshold}**"
        )

        try:
            sources = list_sources()
        except Exception:
            sources = []

        if sources:
            st.markdown(
                '<div class="section-header">Source Inventory</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{len(sources)} sources indexed")
            st.dataframe(
                [
                    {
                        "source": row["source"],
                        "type": row["source_type"],
                        "chunks": row["chunks"],
                        "pages": row["pages"],
                    }
                    for row in sources
                ],
                use_container_width=True,
                hide_index=True,
            )


# ── Main chat interface ────────────────────────────────────────────────────────
def _render_chat():
    # Header
    st.markdown("## 🤖 RAGit")
    st.caption(
        "Multi-Agent RAG · Hybrid Search · Cross-Encoder Reranking · "
        "Hallucination Guard · Grounded Citations"
    )
    st.divider()

    # Render message history
    for msg in st.session_state["messages"]:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

            # Render stored citations if available
            if role == "assistant" and "citations" in msg and msg["citations"]:
                with st.expander("📎 View Sources", expanded=False):
                    _render_citations(msg["citations"])
                    _render_confidence(
                        msg.get("confidence", 0.0),
                        msg.get("refused", False),
                    )

    # Chat input
    if prompt := st.chat_input("Ask me about your documents..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the multi-agent pipeline
        with st.chat_message("assistant"):
            # Pipeline progress indicators
            status_placeholder = st.empty()
            answer_placeholder = st.empty()

            with status_placeholder:
                with st.status("Running multi-agent pipeline...", expanded=True) as status:
                    st.write("🔄 Query Agent — analyzing & rewriting query...")
                    st.write("🔍 Retrieval Agent — hybrid search + reranking...")
                    st.write("✅ Validation Agent — faithfulness check...")
                    st.write("✍️ Generation Agent — building grounded answer...")

                    # Execute pipeline
                    result = run_rag_pipeline(
                        query=prompt,
                        chat_history=st.session_state["messages"][:-1],
                    )
                    status.update(
                        label="✅ Pipeline complete" if not result.get("refused") else "⚠ Insufficient context",
                        state="complete" if not result.get("refused") else "error",
                        expanded=False,
                    )

            # Display the answer
            answer = result.get("answer", "An error occurred.")
            citations = result.get("citations", [])
            confidence = result.get("confidence", 0.0)
            refused = result.get("refused", False)

            answer_placeholder.markdown(answer)

            # Show pipeline debug details in expander
            with st.expander("📊 Pipeline Details", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Faithfulness", f"{result.get('faithfulness_score', 0):.0%}")
                with col2:
                    st.metric("Confidence", f"{confidence:.0%}")
                with col3:
                    st.metric("Sources Found", len(result.get("retrieved_docs", [])))

                if result.get("rewritten_queries"):
                    st.markdown("**Query Reformulations:**")
                    for i, q in enumerate(result["rewritten_queries"], 1):
                        st.markdown(f"  {i}. _{q}_")

                if result.get("validation_reasoning"):
                    st.markdown(f"**Validation:** {result['validation_reasoning']}")

                if result.get("retrieval_debug"):
                    st.markdown("**Retrieval Diagnostics:**")
                    st.dataframe(
                        result["retrieval_debug"],
                        use_container_width=True,
                        hide_index=True,
                    )

            # Citations
            if citations:
                with st.expander("📎 Sources", expanded=True):
                    _render_citations(citations)
                    _render_confidence(confidence, refused)

        # Store in message history (with metadata)
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "confidence": confidence,
            "refused": refused,
        })


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    _init_session()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
