import requests
from domain.models.llm import HealthCheckResp, LLMRequest, LLMResponse, ClearHistoryResponse
from core.log.logger import logger
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from infrastructure.llm.workflow_graph import WorkflowGraph
from infrastructure.llm.loader import clearChatHistory
from core.constants.constants import session_id_key
from infrastructure.rag.retrievers import RetrieverManager
from infrastructure.vector_db.vector_store import VectorStoreManager

class LLMService:
    def __init__(self):
        self.workflow_graph = WorkflowGraph()
        self.vector_store_manager = VectorStoreManager()
        self.retriever_manager = RetrieverManager(self.vector_store_manager)
        
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
                response=resp.get("response", "No response generated"),
                rag_synthesizer_response=resp.get("rag_synthesizer_response", "No RAG synthesizer response generated"),
                rag_routing_decision=resp.get("routing_decision", "No routing decision generated"),
                rag_reflection_comment=resp.get("rag_reflection_comment", "No reflection comment generated")
            )
            
        except Exception as e:
            logger.error(f"[LLM Service Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")