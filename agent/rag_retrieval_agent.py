from typing import List
from langchain_core.documents import Document
from domain.enums.workflow_state import RetrievedDocument, WorkflowState
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from infrastructure.rag.retrievers import RetrieverManager
from infrastructure.vector_db.vector_store import VectorStoreManager
import asyncio


class RAGRetrievalAgent:

    def __init__(self):
        vector_store_manager = VectorStoreManager()
        self.retriever_manager = RetrieverManager(vector_store_manager)

    async def search(self, query_text: str, query_purpose: str) -> List[Document]:
        """
        Execute RAG search for ONE query (for parallel execution).
        This method is used by ToolExecutor for parallel RAG searches.

        Args:
            query_text: The search query
            query_purpose: Description of what this query aims to find

        Returns:
            List of relevant documents from the knowledge base
        """
        try:
            # Run the synchronous retrieval in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                self.retriever_manager.get_relevant_documents,
                query_text
            )

            # Add query metadata to each document
            for doc in documents:
                doc.metadata["search_query"] = query_text
                doc.metadata["query_purpose"] = query_purpose

            return documents

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"RAGRetrievalAgent search error: [{type(e).__name__}]: {str(e)}",
            )

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            if not query:
                raise ValueError("No query found in state")

            documents = self.retriever_manager.get_relevant_documents(query)
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
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"RAGRetrievalAgent error: [{type(e).__name__}]: {str(e)}",
            )
