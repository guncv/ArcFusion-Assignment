from domain.models.information import CheckWorkflowReq, CheckWorkflowResp, HealthCheckResp
from core.log.logger import logger
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from infrastructure.llm.workflow_graph import workflow_graph

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

    # async def invoke_workflow(self, user_input: str) -> str:
    #     try:
    #         result = await self.workflow_graph.invoke(req.user_input)
    #         return result.message
        
    #     except Exception as e:
    #         logger.error(f"[Check Workflow Error]: {e}")
    #         raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
