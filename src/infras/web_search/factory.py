from functools import lru_cache
from typing import Dict, Any, Type, List, Optional
from src.infras.web_search.base import BaseWebSearchProvider
from src.infras.web_search.providers.tavily import TavilyWebSearchProvider
from src.infras import logger
from src.config import config

class WebSearchProviderFactory:
    # Factory for managing web search providers (e.g. Tavily, Bing, etc.)
    _providers: Dict[str, Type[BaseWebSearchProvider]] = {
        "tavily": TavilyWebSearchProvider,
    }

    def __init__(self):
        self._provider_instance: Optional[BaseWebSearchProvider] = None
        
    def get_provider(self) -> BaseWebSearchProvider:
        if self._provider_instance is None:
            web_search_config = config.get("web_search", {})
            if not web_search_config.get("enabled", False):
                raise ValueError("Web search is disabled in configuration")

            provider_name = web_search_config.get("provider", "tavily")

            self._provider_instance = self.create(provider_name)
            logger.info(f"[WebSearchProviderFactory] Provider initialized: {provider_name}")

        return self._provider_instance

    @lru_cache(maxsize=None)
    def create(self, provider_name: str) -> BaseWebSearchProvider:
        provider_name = provider_name.lower()

        if provider_name not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Unsupported web search provider: {provider_name}. "
                f"Available providers: {available}"
            )

        provider_class = self._providers[provider_name]
        provider_instance = provider_class()
        logger.info(f"[WebSearchProviderFactory] Created provider: {provider_name}")
        return provider_instance
