from fastapi import APIRouter
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.services import llm_service
from src.entities import (
    HealthCheckResp,
    LLMRequest,
    LLMResponse,
)

router = APIRouter()

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