from fastapi import FastAPI
from src.routes.chat import chat_router
from src.routes.document import document_router

app = FastAPI(
    title="Chat API"
)

app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(document_router, prefix="/document", tags=["document"])
