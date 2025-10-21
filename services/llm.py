from domain.models.llm import HealthCheckResp, LLMRequest, LLMResponse
from core.log.logger import logger
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from infrastructure.llm import workflow_graph

class LLMService:
    def __init__(self):
        self.workflow_graph = workflow_graph
    
    async def health_check(self) -> HealthCheckResp:
        try:
            resp = HealthCheckResp(
                status="ok"
            )
            return resp
        
        except Exception as e:
            logger.error(f"[Health Check Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")

    async def llm_service(self, req: LLMRequest) -> LLMResponse:
        try:
            result = await self.workflow_graph.invoke(req.user_input)
            logger.info(f"[LLM Service Result]: {result}")
            return LLMResponse(
                message=result.get("response", "An error occurred while processing your request.")
            )
            
        except Exception as e:
            logger.error(f"[LLM Service Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
