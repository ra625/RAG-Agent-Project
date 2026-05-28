from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from rag.embeddings import get_embeddings

VECTORSTORE_DIR = "vectorstore"

def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8")

def ingest(data_dir, chunk_size=300, chunk_overlap=50):
    docs=[]
    for pdf_path in Path(data_dir).glob("*.pdf"):
        pages=PyPDFLoader(str(pdf_path)).load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
            page.page_content = clean_text(page.page_content)

        docs.extend(pages)

    chunks= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                           separators=["\n\n", "\n",". "," ",""],).split_documents(docs)
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=VECTORSTORE_DIR,
        collection_name="pdf_rag",
    )
def load_vectorstore():
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=get_embeddings(),
        collection_name="pdf_rag",
    )

def vectorstore_exists():
    return (Path(VECTORSTORE_DIR)/"chroma.sqlite3").exists()