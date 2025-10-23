from typing import List
from langchain_core.documents import Document
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from infrastructure.rag import get_tavily_web_search

class WebSearchAgent:

    def __init__(self):
        self.web_search = get_tavily_web_search()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")

            if not query:
                raise ValueError("No query found in state")

            web_results: List[Document] = self.web_search.search_as_documents(query)
            logger.info(f"[WebSearchAgent] web results: {web_results}")

            if web_results:
                for i, doc in enumerate(web_results[:3], 1):
                    title = doc.metadata.get("title", "Untitled")
                    url = doc.metadata.get("url", "")
                    logger.debug(f"[WebSearchAgent] Result {i}: {title} - {url}")

            return {
                **state,
                "web_search_results": web_results,
            }

        except Exception as e:
            logger.error(f"[WebSearchAgent] Error during web search: {e}", exc_info=True)

            return {
                **state,
                "web_search_results": [],
            }
