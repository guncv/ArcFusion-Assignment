from fastapi import APIRouter
from src.api.llm import router as llm_router

# Main API router for v1
api_router_v1 = APIRouter()

# Include all sub-routers
api_router_v1.include_router(llm_router, prefix="/llm", tags=["LLM"])
