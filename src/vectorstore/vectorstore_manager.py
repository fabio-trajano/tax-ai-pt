from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def build_faiss_index(chunks: List[Dict], openai_api_key: str):
    """
    Builds a FAISS index from text chunks using OpenAI embeddings.
    """
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    return FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)

def save_faiss_store(vector_store: FAISS, folder_path: str):
    """
    Saves FAISS index to disk.
    """
    vector_store.save_local(folder_path)

def load_faiss_store(folder_path: str):
    """
    Loads FAISS index from disk.
    """
    return FAISS.load_local(folder_path, OpenAIEmbeddings())
