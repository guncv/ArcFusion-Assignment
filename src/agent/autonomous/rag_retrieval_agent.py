from typing import List
from langchain_core.documents import Document
from src.graph.state import RetrievedDocument
from src.graph import WorkflowState
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.infras.retrieval.factory import RetrieverFactory
from src.agent.base import AgentInterface
import asyncio

class RAGRetrievalAgent(AgentInterface):
    def __init__(self):
        self.retriever = RetrieverFactory.get()

    async def search(self, query_text: str, query_purpose: str) -> List[Document]:
        try:
            # Run the synchronous retrieval in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                self.retriever.get_relevant_documents,
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

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            if not query:
                raise ValueError("No query found in state")

            # Run the synchronous retrieval in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                self.retriever.get_relevant_documents,
                query
            )
            
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
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

# Note: rag_retrieval_agent should be instantiated with proper arguments when needed
# rag_retrieval_agent = RAGRetrievalAgent()