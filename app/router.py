from fastapi import APIRouter

from app.api import llm, rag

api_router_v1 = APIRouter()

api_router_v1.include_router(llm.router, prefix="/llm", tags=["LLM"])
api_router_v1.include_router(rag.router, prefix="/rag", tags=["RAG"])