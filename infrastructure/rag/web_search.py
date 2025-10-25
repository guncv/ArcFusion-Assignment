import os
import requests
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from core.config.config import nested_config as config

class TavilyWebSearch:
    def __init__(self):
        self.api_key = config["rag"]["web_search"]["api_key"]
        self.max_results = config["rag"]["web_search"]["max_results"]
        self.search_depth = config["rag"]["web_search"]["search_depth"]
        self.api_url = "https://api.tavily.com/search"

    def search(self, query: str) -> List[Dict[str, Any]]:
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": self.max_results,
                "search_depth": self.search_depth,
            }

            response = requests.post(
                self.api_url,
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
            return []

    def search_as_documents(self,query: str) -> List[Document]:
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
                    "url": url,
                    "title": title,
                    "score": score,
                    "rank": i + 1,
                    "query": query,
                }
            )
            documents.append(doc)

        return documents

    def format_results_for_context(
        self,
        results: List[Dict[str, Any]],
        max_chars_per_result: int = 500
    ) -> str:
        if not results:
            return "No web search results found."

        context_parts = []

        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")

            if len(content) > max_chars_per_result:
                content = content[:max_chars_per_result] + "..."

            formatted = f"""[{i}] {title}
                            URL: {url}
                            {content}
                        """
            context_parts.append(formatted)

        return "\n".join(context_parts)