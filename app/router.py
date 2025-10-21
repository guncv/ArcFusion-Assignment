from fastapi import APIRouter

from app.api import llm

api_router_v1 = APIRouter()

api_router_v1.include_router(llm.router, prefix="/llm", tags=["LLM"])