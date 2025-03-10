from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings

def generate_embeddings_for_chunks(chunks: List[Dict], openai_api_key: str):
    """
    Generates embeddings for text chunks using OpenAI.
    """
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    vectors = embedding_model.embed_documents(texts)

    return [{"embedding": v, "text": t, "metadata": m} for v, t, m in zip(vectors, texts, metadatas)]
