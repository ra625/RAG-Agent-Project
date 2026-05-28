from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers.util import normalize_embeddings


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 model_kwargs={"device": "cpu"},
                                 encode_kwargs={"normalize_embeddings": True},)

