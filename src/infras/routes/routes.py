from fastapi import APIRouter

from src.api import llm_router

api_router_v1 = APIRouter()

api_router_v1.include_router(llm_router, prefix="/llm", tags=["LLM"])