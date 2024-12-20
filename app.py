from typing import List

import streamlit as st
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger

from assistant import get_rag_assistant

st.set_page_config(
    page_title="Local RAG",
    page_icon=":robot:",
)


def restart_assistant():
    st.session_state["rag_assistant"] = None
    st.session_state["rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()


def main() -> None:
    # Get model
    rag_model = st.sidebar.selectbox("Select Model", options=["llama3.2:1b", "llama3:8b", "openhermes", "llama2"])
   
    if "rag_model" not in st.session_state:
        st.session_state["rag_model"] = rag_model
  
    elif st.session_state["rag_model"] != rag_model:
        st.session_state["rag_model"] = rag_model
        restart_assistant()

    rag_assistant: Assistant
    if "rag_assistant" not in st.session_state or st.session_state["rag_assistant"] is None:
        logger.info(f"---*--- Creating {rag_model} Assistant ---*---")
        rag_assistant = get_rag_assistant(model=rag_model)
        st.session_state["rag_assistant"] = rag_assistant
    else:
        rag_assistant = st.session_state["rag_assistant"]

    try:
        st.session_state["rag_assistant_run_id"] = rag_assistant.create_run()
    except Exception:
        st.warning("Could not create assistant, is the database running?")
        return

    assistant_chat_history = rag_assistant.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = assistant_chat_history
    else:
        logger.debug("No chat history found")
        st.session_state["messages"] = [{"role": "assistant", "content": "Upload a file or ask me questions, how can I help you?"}]

    if prompt := st.chat_input(placeholder="Ask me regarding the document..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})

    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            index=0
            try:
                
                for delta in rag_assistant.run(question):
                    # delta = delta.replace("{", "\\\{").replace("}", "\\}")
                    # delta = delta.replace(";","\\\;").replace(";", "\\\;")
                    
                    # response+=str(index)+
                    # logger.debug(f"{index}Delta:+ "+str(delta))
                    # delta = delta.replace("{", "\\\{").replace("}", "\\}")
                    index+=1
                    response="".join([response,str(delta)])  # type: ignore
                    
                    resp_container.markdown(response)
                # resp_container.markdown(response)
            except Exception as e:
                response = f"Error: {e}"
                resp_container.markdown(response)
                
                # 
            st.session_state["messages"].append({"role": "assistant", "content": response})
            logger.debug(f"Assistant response: {response}")

    if rag_assistant.knowledge_base:

        if "url_scrape_key" not in st.session_state:
            st.session_state["url_scrape_key"] = 0

        input_url = st.sidebar.text_input(
            "Add URL to Knowledge Base", type="default", key=st.session_state["url_scrape_key"]
        )
        add_url_button = st.sidebar.button("Add URL")
        if add_url_button:
            if input_url is not None:
                alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
                if f"{input_url}_scraped" not in st.session_state:
                    scraper = WebsiteReader(max_links=2, max_depth=1)
                    web_documents: List[Document] = scraper.read(input_url)
                    logger.debug(f"Scraped {len(web_documents)} documents from {input_url}")
                    logger.debug(web_documents)
                    if web_documents:
                        logger.debug("Adding documents to knowledge base")
                        rag_assistant.knowledge_base.load_documents(web_documents, upsert=True)
                    else:
                        st.sidebar.error("Could not read website")
                    st.session_state[f"{input_url}_uploaded"] = True
                alert.empty()

        # Add PDFs to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 100

        uploaded_file = st.sidebar.file_uploader(
            "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
        )
        if uploaded_file is not None:
            alert = st.sidebar.info("Processing PDF...", icon="🧠")
            rag_name = uploaded_file.name.split(".")[0]
            if f"{rag_name}_uploaded" not in st.session_state:
                reader = PDFReader()
                rag_documents: List[Document] = reader.read(uploaded_file)
                logger.debug(f"Read {len(rag_documents)} documents from {uploaded_file.name}")
                logger.debug(rag_documents)
                if rag_documents:
                    
                    rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
                    
                else:
                    st.sidebar.error("Could not read PDF")
                st.session_state[f"{rag_name}_uploaded"] = True
            alert.empty()

    if rag_assistant.knowledge_base and rag_assistant.knowledge_base.vector_db:
        if st.sidebar.button("Clear Knowledge Base"):
            rag_assistant.knowledge_base.vector_db.delete()
            logger.info("Knowledge base cleared")
            st.sidebar.success("Knowledge base cleared")

    if rag_assistant.storage:
        rag_assistant_run_ids: List[str] = rag_assistant.storage.get_all_run_ids()
        new_rag_assistant_run_id = st.sidebar.selectbox("Run ID", options=rag_assistant_run_ids)
        if st.session_state["rag_assistant_run_id"] != new_rag_assistant_run_id:
            logger.info(f"---*--- Loading {rag_model} run: {new_rag_assistant_run_id} ---*---")
            st.session_state["rag_assistant"] = get_rag_assistant(model=rag_model, run_id=new_rag_assistant_run_id)
            st.rerun()

    if st.sidebar.button("New Run"):
        restart_assistant()


main()
