import tempfile
import streamlit as st
from pathlib import Path
from rag.ingest import  ingest, load_vectorstore, vectorstore_exists
from rag.chain import  build_rag_chain

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Ask your PDFs")

uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Ingest PDFs"):
    with st.spinner("Processing..."):
        data_dir=Path(tempfile.mkdtemp())
        for f in uploaded_files:
            (data_dir/f.name).write_bytes(f.read())
            vs = ingest(str(data_dir))
            st.session_state.vectorstore = vs
            st.session_state.rag_chain=build_rag_chain(vs)
            st.session_state_history=[]
            st.success(f"Ingested {len(uploaded_files)} PDF(s)!")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])

if st.session_state.vectorstore:
    question = st.chat_input("Ask Something... ")
    if question:
        with st.spinner("Thinking... "):
            result = st.session_state.rag_chain(question)
            st.session_state.history.append({
                "question": question,
                "answer": result["answer"],
            })
            st.rerun()

    else:
        st.info("Upload and ingest PDFs to get started")

