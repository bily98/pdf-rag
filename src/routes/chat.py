from fastapi import APIRouter
from src.services.chat import get_llm_status, ask_question

chat_router = APIRouter()


@chat_router.get("/status")
async def get_status():
    answer = get_llm_status()
    return {"chat": answer}

@chat_router.post("/ask")
async def post_chat(question: str):
    answer = ask_question(question=question)
    return {"chat": answer}