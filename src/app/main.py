import os
from dotenv import load_dotenv

from src.ingestion.extract_and_chunk import load_and_split_pdfs
from src.embeddings.generate_embeddings import generate_embeddings_for_chunks
from src.vectorstore.vectorstore_manager import (
    build_faiss_index,
    save_faiss_store,
    load_faiss_store
)
from src.chains.qa_chain import create_qa_chain

def main():
    # Load environment variables from .env
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "../../data")
    faiss_store_folder = os.path.join(current_dir, "../../faiss_index")

    # Check if FAISS index already exists
    if not os.path.exists(faiss_store_folder):
        print("No existing FAISS index found. Creating a new one...")
        # 1) Load and split PDFs
        chunks = load_and_split_pdfs(data_folder)

        # 2) Generate embeddings
        embedded_data = generate_embeddings_for_chunks(chunks, openai_api_key)

        # 3) Build FAISS index and save it
        vector_store = build_faiss_index(embedded_data)
        save_faiss_store(vector_store, faiss_store_folder)
    else:
        print("Loading existing FAISS index...")
        vector_store = load_faiss_store(faiss_store_folder)

    # Create the Q&A chain with our vector store and OpenAI
    qa_chain = create_qa_chain(vector_store, openai_api_key)

    print("\n=== Portugues Tax AI Chatbot ===")
    print("Ask a question about taxes in Portugal. Type 'exit' or 'quit' to stop.")

    while True:
        user_question = input("\nQuestion: ")
        if user_question.lower() in ["exit", "quit", "sair"]:
            print("Exiting. Goodbye!")
            break

        response = qa_chain.run(user_question)
        print(f"\nAnswer:\n{response}")

if __name__ == "__main__":
    main()
