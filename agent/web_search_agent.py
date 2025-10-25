from typing import List
from langchain_core.documents import Document
from domain.enums.workflow_state import WorkflowState
from infrastructure.rag import get_tavily_web_search
import asyncio

class WebSearchAgent:
    def __init__(self):
        self.web_search = get_tavily_web_search()

    async def invoke(self, query_text: str, query_purpose: str) -> List[Document]:
        try:
            # Run the synchronous search in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            results: List[Document] = await loop.run_in_executor(
                None, 
                self.web_search.search_as_documents, 
                query_text
            )

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
            return []
