import asyncio
from typing import List
from langchain_core.documents import Document
from src.graph.state import RetrievedDocument
from src.graph import WorkflowState
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes, LLMAgentName
from src.infras.retrieval.factory import RetrieverFactory
from src.infras.web_search.factory import web_search_factory
from src.agent.base import AgentInterface
from src.infras.log import logger

class HybridRetrievalAgent(AgentInterface):

    def __init__(self):
        self.rag_retriever = RetrieverFactory.get()
        self.web_search = web_search_factory.get_provider()
        self._agent_name = LLMAgentName.HYBRID_RETRIEVAL_AGENT.value

    @property
    def name(self) -> str:
        return self._agent_name or super().name

    async def _retrieve_rag(self, query: str) -> List[RetrievedDocument]:
        try:
            # Run synchronous retrieval in thread pool
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                self.rag_retriever.get_relevant_documents,
                query
            )

            retrieved_documents_with_scores = []
            for doc in documents:
                score = doc.metadata.get("score", 0.0)
                retrieved_documents_with_scores.append(RetrievedDocument(
                    document=doc,
                    score=score
                ))

            logger.info(f"HybridRetrievalAgent RAG: retrieved {len(retrieved_documents_with_scores)} documents")
            return retrieved_documents_with_scores
        except Exception as e:
            logger.error(f"HybridRetrievalAgent RAG error: {str(e)}")
            return []

    async def _retrieve_web(self, query: str) -> List[Document]:
        try:
            # Run synchronous search in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.web_search.search_as_documents,
                query
            )

            logger.info(f"HybridRetrievalAgent Web: retrieved {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"HybridRetrievalAgent Web error: {str(e)}")
            return []

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            # Get separate queries for RAG and Web (optimized for each source)
            rag_query = state.get("rag_query", "")
            web_query = state.get("web_query", "")

            # Fallback to generated_queries or user_query if dual queries not provided
            if not rag_query or not web_query:
                fallback_query = state.get("generated_queries") or state.get("user_query", "")
                rag_query = rag_query or fallback_query
                web_query = web_query or fallback_query

            if not rag_query or not web_query:
                raise ValueError("No queries found in state for hybrid search")

            logger.info(
                f"HybridRetrievalAgent executing parallel search:\n"
                f"  RAG query: '{rag_query}'\n"
                f"  Web query: '{web_query}'"
            )

            rag_task = self._retrieve_rag(rag_query)
            web_task = self._retrieve_web(web_query)

            retrieved_documents_with_scores, web_search_results = await asyncio.gather(
                rag_task,
                web_task,
                return_exceptions=False
            )

            logger.info(
                f"HybridRetrievalAgent completed: "
                f"RAG={len(retrieved_documents_with_scores)} docs, "
                f"Web={len(web_search_results)} results"
            )

            return {
                **state,
                "retrieved_documents_with_scores": retrieved_documents_with_scores,
                "web_search_results": web_search_results,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )
