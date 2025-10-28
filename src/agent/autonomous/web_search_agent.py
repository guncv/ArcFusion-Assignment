from typing import List
from langchain_core.documents import Document
from src.graph import WorkflowState
from src.constants import LLMAgentName
from src.infras.web_search.factory import web_search_factory
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import AgentInterface
from src.infras.log import logger

class WebSearchAgent(AgentInterface):
    def __init__(self):
        self.web_search = web_search_factory.get_provider()
        self._agent_name = LLMAgentName.WEB_SEARCH_AGENT.value

    @property
    def name(self) -> str:
        return self._agent_name or super().name

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

            logger.info(f"WebSearchAgent executing query: '{query_text}'")

            # Direct synchronous call - single search query
            results: List[Document] = self.web_search.search_as_documents(query_text)

            logger.info(f"WebSearchAgent retrieved: {results}")

            return {
                **state,
                "web_search_results": results,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

