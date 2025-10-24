from fastapi import APIRouter
from fastapi.responses import JSONResponse
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from core.log.logger import logger
from domain.models.rag import RAGQueryRequest
from services.llm import LLMService
from domain.models.llm import HealthCheckResp, LLMRequest, LLMResponse, ClearHistoryResponse

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
        
@router.post("/", response_model=LLMResponse)
async def llm_api(req: LLMRequest) -> LLMResponse:
    try:
        resp = await llm_service.llm_service(req)
        return resp
    except (ArcFusionException, Exception) as e:
        if type(e) != ArcFusionException:
            e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        logger.error(f"[LLM API Error]: {e}")
        e.raise_HTTPException()

@router.post("/clear-history")
async def clear_history_api():
    try:
        resp = await llm_service.clear_chat_history()
        return resp
    except (ArcFusionException, Exception) as e:
        if type(e) != ArcFusionException:
            e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        logger.error(f"[Clear History API Error]: {e}")
        e.raise_HTTPException()
        
@router.post("/test-rag-retrieval")
async def test_rag_retrieval_api(req: RAGQueryRequest) -> JSONResponse:
    try:
        resp = await llm_service.test_rag_retrieval(req.query)
        return JSONResponse(content=resp)
    except (ArcFusionException, Exception) as e:
        if type(e) != ArcFusionException:
            e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        logger.error(f"[Test RAG Retrieval API Error]: {e}")
        e.raise_HTTPException()