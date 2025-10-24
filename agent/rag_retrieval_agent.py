from typing import List
from langchain_core.documents import Document
from core.log.logger import logger
from domain.enums.workflow_state import RetrievedDocument, WorkflowState
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from infrastructure.rag.retrievers import RetrieverManager
from infrastructure.vector_db.vector_store import VectorStoreManager


class RAGRetrievalAgent:

    def __init__(self):
        vector_store_manager = VectorStoreManager()
        self.retriever_manager = RetrieverManager(vector_store_manager)

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            if not query:
                raise ValueError("No query found in state")

            documents = self.retriever_manager.get_relevant_documents(query)
            logger.info(f"[RAGRetrievalAgent] retrieved documents: {len(documents)}")

            retrieved_documents_with_scores = []
            for doc in documents:
                score = doc.metadata.get("score", 0.0)
                retrieved_documents_with_scores.append(RetrievedDocument(
                    document=doc,
                    score=score
                ))
                
            return {
                **state,
                "retrieved_documents_with_scores": retrieved_documents_with_scores,
            }

        except Exception as e:
            logger.error(f"[RAGRetrievalAgent] Error during retrieval: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"RAGRetrievalAgent error: [{type(e).__name__}]: {str(e)}",
            )
