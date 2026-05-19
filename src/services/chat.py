import os
from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama
from src.database.pinecone import get_vector_stores

load_dotenv()

def get_llm():
    llm = Ollama(model=os.environ['LLAMA_MODEL'])

    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversational_chain():
    llm = get_llm()
    vectordb = get_vector_stores()
    retriever = vectordb.as_retriever(search_kwargs={"k": 20})

    template = """Usa el siguiente contexto para dar una respuesta concisa a la pregunta. Si no conoces la respuesta unicamente di "No se" y no intentes dar una respuesta generalizada. 
    {context}
    Pregunta: {input}"""

    retrieval_qa_chat_prompt = PromptTemplate.from_template(template)#hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain

def get_llm_status():
    conversational_chain = get_conversational_chain()

    result = conversational_chain.invoke({"input": '¿Quién eres?'})
    
    return result['answer']

def ask_question(question):
    conversational_chain = get_conversational_chain()

    result = conversational_chain.invoke({'input': question})
    
    return result