from typing import List
from langchain_core.documents import Document
from src.entities import (
    HealthCheckResp,
    LLMRequest,
    LLMResponse,
    ClearHistoryResponse
)
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.graph import workflow_graph
from src.constants import SESSION_ID_KEY
from src.infras.web_search.factory import web_search_factory

class LLMService:
    def __init__(self):
        self.workflow_graph = workflow_graph
        self.web_search_tool = web_search_factory.get_provider()
        
    async def health_check(self) -> HealthCheckResp:
        try:
            resp = HealthCheckResp(
                status="ok"
            )
            return resp
        
        except Exception as e:
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")

    async def llm_service(self, req: LLMRequest) -> LLMResponse:
        try:
            resp = await self.workflow_graph.ainvoke(req.user_input, SESSION_ID_KEY)

            return LLMResponse(
                response=resp.get("response", "No response generated")
            )
            
        except Exception as e:
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        
    async def clear_chat_history(self) -> ClearHistoryResponse:
        try:
            self.graph.clear_chat_history()
            return ClearHistoryResponse(message="Chat history cleared successfully")
        except Exception as e:
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        
    async def web_search(self, query: str) -> List[Document]:
        try:
            results = self.web_search_tool.search_as_documents(query)
            return results
        except Exception as e:
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        
llm_service = LLMService()