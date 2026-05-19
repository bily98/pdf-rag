from fastapi import APIRouter, File, UploadFile
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from src.database.pinecone import set_documents

document_router = APIRouter()

@document_router.post("/")
async def post_document(file: UploadFile = File(...)):
    contents = file.file.read()

    temp_file_path = f"./{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(contents)

    print("Loading documents from PDF...")
    loader = UnstructuredFileLoader(temp_file_path)
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400
    )
    texts = text_splitter.split_documents(documents)

    print("Uploading documents to Pinecone...")
    set_documents(texts)

    return {"message": "Document uploaded successfully!"}