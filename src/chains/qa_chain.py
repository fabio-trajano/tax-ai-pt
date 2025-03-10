from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

def create_custom_qa_chain(vector_store, openai_api_key: str):
    """
    Creates a Q&A chain with a custom prompt, forcing the model
    to rely ONLY on the provided text chunks from the vector store.
    """
    template = """Você é um assistente especializado em legislação fiscal portuguesa.
Se baseie EXCLUSIVAMENTE no texto dos documentos fornecidos para responder.
Se não encontrar uma resposta na legislação, diga: "Não encontrei referência específica na legislação."

Texto de contexto:
{context}

Pergunta: {question}

Resposta:"""

    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    return qa_chain
