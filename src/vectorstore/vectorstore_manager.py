# src/vectorstore/vectorstore_manager.py

import faiss
from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def build_faiss_index(embedded_data: List[Dict], openai_api_key: str):
    """
    Creates a FAISS index from a list of dicts that contain
    'embedding', 'text', and 'metadata'. Returns a LangChain
    FAISS vectorstore object.

    :param embedded_data: List of dictionaries, each with:
        - 'embedding': List[float] (precomputed vector)
        - 'text': str (original text)
        - 'metadata': dict (source file, page, etc.)
    :param openai_api_key: API key to instantiate an Embeddings model
    :return: FAISS vectorstore object
    """
    # Extract lists from embedded_data
    embeddings = [item["embedding"] for item in embedded_data]  # List[List[float]]
    texts = [item["text"] for item in embedded_data]            # List[str]
    metadatas = [item["metadata"] for item in embedded_data]    # List[dict]

    # Create a fresh FAISS index (dimension = size of the embedding vectors)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    # Initialize an Embeddings object (needed by FAISS.from_embeddings)
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Use keyword arguments to match the current signature of FAISS.from_embeddings
    # We must pass an 'embedding' object plus optional lists for texts, metadatas, and an existing faiss_index
    vector_store = FAISS.from_embeddings(
        embeddings=embeddings,
        embedding=embedding_model,
        metadatas=metadatas,
        texts=texts,
        faiss_index=index
    )

    return vector_store

def save_faiss_store(vector_store: FAISS, folder_path: str):
    """
    Saves the FAISS index and metadata to disk so it can be reloaded later.
    """
    vector_store.save_local(folder_path)

def load_faiss_store(folder_path: str):
    """
    Loads a FAISS vectorstore from a local folder, re-initializing it
    with the same embedding model signature (OpenAIEmbeddings).

    :param folder_path: Path where the index was saved
    :return: A LangChain FAISS vectorstore object
    """
    return FAISS.load_local(folder_path, OpenAIEmbeddings())
