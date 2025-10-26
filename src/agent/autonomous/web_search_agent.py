from typing import List
from langchain_core.documents import Document
from src.graph import WorkflowState
from src.constants import LLMAgentName
from src.infras.web_search.factory import web_search_factory
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
import asyncio
from src.agent.base import agent_interface

class WebSearchAgent(agent_interface):
    def __init__(self):
        self.web_search = web_search_factory.get_provider()
        self._agent_name = LLMAgentName.WEB_SEARCH_AGENT.value

    @property
    def name(self) -> str:
        return self._agent_name or super().name

    async def ainvoke(self, query_text: str, query_purpose: str) -> List[Document]:
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
            # Return empty list on error rather than failing
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

