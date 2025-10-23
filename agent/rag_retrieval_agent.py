from typing import List
from langchain_core.documents import Document
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from infrastructure.rag import get_rag_pipeline
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes


class RAGRetrievalAgent:

    def __init__(self):
        self.rag_pipeline = get_rag_pipeline()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            if not query:
                raise ValueError("No query found in state")

            retrieval_results = self.rag_pipeline.retrieve(query)
            documents: List[Document] = retrieval_results.get("local_docs", [])
            web_docs: List[Document] = retrieval_results.get("web_docs", [])
            
            logger.info(f"[RAGRetrievalAgent] local documents: {len(documents)}, web documents: {len(web_docs)}")

            scores = []
            for doc in documents:
                score = doc.metadata.get("relevance_score", 0.0)
                scores.append(score)
                
            return {
                **state,
                "retrieved_documents": documents,
                "web_documents": web_docs,
                "retrieval_scores": scores,
            }

        except Exception as e:
            logger.error(f"[RAGRetrievalAgent] Error during retrieval: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"RAGRetrievalAgent error: [{type(e).__name__}]: {str(e)}",
            )
