import streamlit as st
import os
import tempfile
from ingest import ingest_pdf, clear_database
from retrieval import get_hybrid_results, rerank_results, generate_answer, condense_query

st.set_page_config(page_title="Hybrid-RAG Support Desk", layout="wide", initial_sidebar_state="expanded")

st.title("Hybrid-RAG Support Desk")

with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox("Select LLM", ["Gemini", "Claude", "GPT-4o"])
    api_key = st.text_input(f"{model_choice} API Key", type="password")
    
    if model_choice == "Gemini":
        os.environ["GEMINI_API_KEY"] = api_key
    elif model_choice == "Claude":
        os.environ["ANTHROPIC_API_KEY"] = api_key
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a PDF Policy Document", type=["pdf"])
    
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process Document", use_container_width=True)
    with col2:
        clear_btn = st.button("Reset Database", use_container_width=True)
        
    if clear_btn:
        clear_database()
        st.session_state.messages = []
        st.success("Database cleared!")

    if process_btn:
        if uploaded_file is not None:
            with st.spinner("Ingesting document (Extracting, Chunking, Embedding, Indexing)..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    ingest_pdf(tmp_path)
                    st.success("Document ingested successfully!")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        else:
            st.warning("Please upload a PDF first.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about the uploaded policies..."):
    if not api_key:
        st.error(f"Please provide your {model_choice} API Key in the sidebar.")
    else:
        chat_history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                try:
                    search_query = query
                    if chat_history:
                        search_query = condense_query(query, chat_history, model_choice)
                        if search_query != query:
                            st.caption(f"*(Rewrote query for context: {search_query})*")
                            
                    hybrid_chunks = get_hybrid_results(search_query, top_k=20)
                    
                    if not hybrid_chunks:
                        st.warning("No documents found. Please upload and process a document first.")
                    else:
                        top_5_chunks = rerank_results(search_query, hybrid_chunks, top_n=5)
                        answer = generate_answer(search_query, top_5_chunks, model_choice)
                        
                        st.markdown(answer)
                        with st.expander("View Retrieved Context (Top 5)"):
                            for i, chunk in enumerate(top_5_chunks):
                                st.markdown(f"**Chunk {i+1}**\n\n{chunk}")
                                st.divider()
                                
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
