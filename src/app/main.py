import os
from dotenv import load_dotenv
from src.chains.qa_chain import create_custom_qa_chain
from src.vectorstore.vectorstore_manager import build_faiss_index, save_faiss_store, load_faiss_store
from src.ingestion.extract_and_chunk import load_and_split_pdfs

def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "../../data")
    faiss_store_folder = os.path.join(current_dir, "../../faiss_index")

    if not os.path.exists(faiss_store_folder):
        print("No existing FAISS index found. Creating a new one...")
        chunks = load_and_split_pdfs(data_folder)
        vector_store = build_faiss_index(chunks, openai_api_key)
        save_faiss_store(vector_store, faiss_store_folder)
    else:
        print("Loading existing FAISS index...")
        vector_store = load_faiss_store(faiss_store_folder)

    qa_chain = create_custom_qa_chain(vector_store, openai_api_key)

    print("\n=== TaxAI Chatbot ===")
    print("Ask a question about taxes. Type 'exit' or 'quit' to stop.")

    while True:
        user_question = input("\nQuestion: ")
        if user_question.lower() in ["exit", "quit", "sair"]:
            print("Exiting. Goodbye!")
            break

        response = qa_chain.run(user_question)
        print(f"\nAnswer:\n{response}")

if __name__ == "__main__":
    main()
