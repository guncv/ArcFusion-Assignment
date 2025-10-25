from typing import List
from fastapi import APIRouter
from langchain_core.documents import Document
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from services.llm import LLMService
from domain.models.llm import HealthCheckResp, LLMRequest, LLMResponse
from domain.models.llm import WebSearchRequest

router = APIRouter()

llm_service = LLMService()

@router.get("/health-check", response_model=HealthCheckResp)
async def health_check_api() -> HealthCheckResp:
    try:
        resp = await llm_service.health_check()
        return resp
    except (ArcFusionException, Exception) as e:
        if type(e) != ArcFusionException:
            e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        e.raise_HTTPException()
        
@router.post("/", response_model=LLMResponse)
async def llm_api(req: LLMRequest) -> LLMResponse:
    try:
        resp = await llm_service.llm_service(req)
        return resp
    except (ArcFusionException, Exception) as e:
        if type(e) != ArcFusionException:
            e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        e.raise_HTTPException()

@router.post("/clear-history")
async def clear_history_api():
    try:
        resp = await llm_service.clear_chat_history()
        return resp
    except (ArcFusionException, Exception) as e:
        if type(e) != ArcFusionException:
            e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        e.raise_HTTPException()
                
@router.post("/web-search", response_model=List[Document])
async def web_search_api(req: WebSearchRequest) -> List[Document]:
    try:
        resp = await llm_service.web_search(req.query)
        return resp
    except (ArcFusionException, Exception) as e:
        if type(e) != ArcFusionException:
            e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        e.raise_HTTPException()