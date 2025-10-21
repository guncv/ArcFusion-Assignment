from fastapi import APIRouter
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from core.log.logger import logger
from services.llm import LLMService
from domain.models.information import HealthCheckResp

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
        logger.error(f"[Health Check API Error]: {e}")
        e.raise_HTTPException()