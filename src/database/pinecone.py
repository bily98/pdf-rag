import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name = os.environ['PINECONE_INDEX_NAME']

def get_index():
    if index_name not in pc.list_indexes().names():
        pc.create_index(
        name=index_name,
        dimension=768,
        spec=ServerlessSpec(
            cloud=os.environ['PINECONE_CLOUD'],
            region=os.environ['PINECONE_REGION']
        )
    )
        
    index = pc.Index(index_name)

    return index

def get_vector_stores():
    index = get_index()
    
    embeddings = HuggingFaceEmbeddings()
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    return vector_store

def set_documents(documents):
    vector_store = get_vector_stores()

    vector_store.add_documents(documents=documents)