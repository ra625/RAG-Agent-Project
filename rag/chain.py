import streamlit  as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from rag.retriever import get_retriever, format_citations

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant that answers questions strictly on the provided context.
Context:{context}

Question:{question}

Answer based ONLY on the context.If the context lacks enough information, say so. """
)

def format_docs(docs):
    parts= []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unkown")
        page = doc.metadata.get("page", "?")
        page_display=int(page)+1 if isinstance(page, int) else page
        parts.append(f"[{i}]{source}, Page {page_display}\n{doc.page_content}")
    return "\n\n".join(parts)

def  build_rag_chain(vectorstore, k=5):
    retriever = get_retriever(vectorstore, k=k )
    llm =  ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=st.secrets["LLM_API_KEY"],
    )

    answer_chain=(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(), }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    def run(question):
        docs = retriever.invoke(question)
        return {
            "answer": answer_chain.invoke(question) ,
            "sources": format_citations(docs),
        }
    return run
