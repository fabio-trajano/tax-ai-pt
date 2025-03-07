from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings


def generate_embeddings_for_chunks(chunks: List[Dict], openai_api_key: str):
    """
    Receives a list of dictionaries, each containing 'text' and 'metadata',
    uses OpenAI to generate embeddings, and returns a list of dicts with
    'embedding', 'text', and 'metadata'.
    """
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Batch-generate embeddings
    vectors = embedding_model.embed_documents(texts)

    embedded_data = []
    for vector, text, meta in zip(vectors, texts, metadatas):
        embedded_data.append({
            "embedding": vector,
            "text": text,
            "metadata": meta
        })

    return embedded_data
