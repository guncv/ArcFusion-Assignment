from typing import List
from langchain_core.documents import Document
from src.graph.state import RetrievedDocument
from src.graph import WorkflowState
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.infras.retrieval.factory import RetrieverFactory
from src.agent.base import AgentInterface

class RAGRetrievalAgent(AgentInterface):
    def __init__(self):
        self.retriever = RetrieverFactory.get()

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            # Get the query from generated_queries
            generated_queries = state.get("generated_queries", "")

            if not generated_queries:
                # Fallback to user_query if no generated queries
                query_text = state.get("user_query", "")
            else:
                # Use planner-generated query
                query_text = generated_queries

            if not query_text:
                raise ValueError("No query found in state")

            from src.infras.log import logger
            logger.info(f"RAGRetrievalAgent executing query: '{query_text}'")

            # Direct synchronous call - since we're only doing a single RAG retrieval
            documents = self.retriever.get_relevant_documents(query_text)

            retrieved_documents_with_scores = []
            for doc in documents:
                score = doc.metadata.get("score", 0.0)
                retrieved_documents_with_scores.append(RetrievedDocument(
                    document=doc,
                    score=score
                ))

            logger.info(f"RAGRetrievalAgent retrieved: {retrieved_documents_with_scores}")

            return {
                **state,
                "retrieved_documents_with_scores": retrieved_documents_with_scores,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )