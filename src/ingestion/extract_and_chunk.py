import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdfs(pdf_folder: str, chunk_size=1000, chunk_overlap=100):
    """
    Loads PDFs from a folder, extracts text, and splits it into chunks.
    """
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        for doc in loader.load():
            for chunk in text_splitter.split_text(doc.page_content):
                chunks.append({"text": chunk, "metadata": {"source_file": pdf_file}})

    return chunks
