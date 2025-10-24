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
        
    async def test_rag_retrieval(self, query: str) -> dict:
        try:
            documents = self.retriever_manager.get_relevant_documents(query)
            
            logger.info(f"[Test RAG Retrieval] Retrieved documents: {documents}")
            serializable_docs = []
            for doc in documents:
                serializable_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return {
                "query": "latest trends in LangGraph multi-agent orchestration",
                "documents": serializable_docs,
                "document_count": len(serializable_docs)
            }
        except Exception as e:
            logger.error(f"[Test RAG Retrieval Error]: {e}")
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")