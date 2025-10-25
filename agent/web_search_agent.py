from typing import List
from langchain_core.documents import Document
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from infrastructure.rag import get_tavily_web_search

class WebSearchAgent:
    def __init__(self):
        self.web_search = get_tavily_web_search()

    async def invoke(self, query_text: str, query_purpose: str) -> List[Document]:
        try:
            results: List[Document] = self.web_search.search_as_documents(query_text)

            # Filter results by relevance score (> 0.6)
            filtered_results = []
            for doc in results:
                score = doc.metadata.get("score", 0.0)
                if score > 0.6:
                    doc.metadata["search_query"] = query_text
                    doc.metadata["query_purpose"] = query_purpose
                    filtered_results.append(doc)
                    
            return filtered_results

        except Exception as e:
            logger.error(f"[WebSearchAgent] Error during web search query: {query_text}: {e}", exc_info=True)

            return []
