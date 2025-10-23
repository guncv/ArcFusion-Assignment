import requests
from domain.models.llm import HealthCheckResp, LLMRequest, LLMResponse, ClearHistoryResponse
from core.log.logger import logger
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from infrastructure.llm.workflow_graph import WorkflowGraph
from infrastructure.llm.loader import clearChatHistory
from core.constants.constants import session_id_key

class LLMService:
    def __init__(self):
        self.workflow_graph = WorkflowGraph()
    
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
            resp = await self.workflow_graph.invoke(req.user_input, session_id_key)

            return LLMResponse(
                message=resp.get("response", "No response generated")
            )

        except Exception as e:
            logger.error(f"[LLM Service Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")

    async def clear_chat_history(self):
        try:
            success = clearChatHistory(session_id_key)
            if success:
                return
            else:
                raise ArcFusionException(
                    error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                    description="Failed to clear chat history"
                )

        except ArcFusionException:
            raise
        except Exception as e:
            logger.error(f"[Clear History Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        
    async def test_web_search(self) -> str:
        try:
            API_KEY = "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"
            query = "latest trends in LangGraph multi-agent orchestration"

            resp = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "query": query,
                    "max_results": 5,
                    "include_domains": [],
                    "search_depth": "advanced"
                }
            )

            data = resp.json()
            return data

        except Exception as e:
            logger.error(f"[Test Web Search Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")