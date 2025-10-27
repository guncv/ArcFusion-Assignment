import requests
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.infras.web_search import BaseWebSearchProvider
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.config import config

class TavilyWebSearchProvider(BaseWebSearchProvider):

    def __init__(self):
        tavily_config = config.get("web_search", {}).get("tavily", {})
        self._api_key = tavily_config.get("api_key")
        self._max_results = tavily_config.get("max_results", 3)
        self._search_depth = tavily_config.get("search_depth", "advanced")
        self._api_url = tavily_config.get("api_url", "https://api.tavily.com/search")

    def search(self, query: str) -> List[Dict[str, Any]]:
        try:
            payload = {
                "api_key": self._api_key,
                "query": query,
                "max_results": self._max_results,
                "search_depth": self._search_depth,
            }

            response = requests.post(
                self._api_url,
                headers={
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10
            )

            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            return results
        except Exception as e:
            raise ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
            

    def search_as_documents(self, query: str) -> List[Document]:
        results = self.search(query)

        documents = []
        for i, result in enumerate(results):
            # Extract content and metadata
            content = result.get("content", "")
            url = result.get("url", "")
            title = result.get("title", "")
            score = result.get("score", 0.0)

            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": "web_search",
                    "provider": "tavily",
                    "url": url,
                    "title": title,
                    "score": score,
                    "rank": i + 1,
                    "query": query,
                }
            )
            documents.append(doc)

        return documents

