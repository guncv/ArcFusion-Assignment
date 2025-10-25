from typing import List
from langchain_core.documents import Document
import requests
from domain.models.llm import HealthCheckResp, LLMRequest, LLMResponse, ClearHistoryResponse
from core.log.logger import logger
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from infrastructure.llm.workflow_graph import WorkflowGraph
from core.constants.constants import session_id_key
from infrastructure.rag import get_tavily_web_search

class LLMService:
    def __init__(self):
        self.workflow_graph = WorkflowGraph()
        self.web_search_tool = get_tavily_web_search()
        
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
                response=resp.get("response", "No response generated")
            )
            
        except Exception as e:
            logger.error(f"[LLM Service Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        
    async def clear_chat_history(self) -> ClearHistoryResponse:
        try:
            self.workflow_graph.clear_chat_history()
            return ClearHistoryResponse(message="Chat history cleared successfully")
        except Exception as e:
            logger.error(f"[Clear Chat History Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        
    async def web_search(self, query: str) -> List[Document]:
        try:
            results = self.web_search_tool.search_as_documents(query)
            return results
        except Exception as e:
            logger.error(f"[Web Search Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")