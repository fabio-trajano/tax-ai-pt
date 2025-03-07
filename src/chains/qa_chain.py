from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def create_qa_chain(vector_store, openai_api_key: str):
    """
    Creates a LangChain Q&A chain with a Retriever from the vector store
    and an OpenAI LLM (GPT-3.5).
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # We'll retrieve the top 3 most similar chunks
    )

    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo"
    )

    # "stuff" is a simple chain_type that concatenates all retrieved chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain
