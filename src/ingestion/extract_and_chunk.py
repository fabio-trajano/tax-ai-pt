import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdfs(pdf_folder: str, chunk_size=1000, chunk_overlap=100):
    """
    Loads all PDFs from a folder, extracts text, and splits it into chunks.
    Returns a list of dictionaries, each containing 'text' and 'metadata'.
    """
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        for doc in documents:
            splits = text_splitter.split_text(doc.page_content)
            for chunk in splits:
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source_file": pdf_file
                    }
                })

    return all_chunks
