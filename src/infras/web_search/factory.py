from functools import lru_cache
from typing import Dict, Type
from src.infras.web_search.base import BaseWebSearchProvider
from src.infras.web_search.providers.tavily import TavilyWebSearchProvider
from src.infras import logger
from src.config import config

class WebSearchProviderFactory:
    _providers: Dict[str, Type[BaseWebSearchProvider]] = {
        "tavily": TavilyWebSearchProvider,
    }

    @classmethod
    @lru_cache(maxsize=1)
    def get_provider(cls) -> BaseWebSearchProvider:
        web_search_config = config.get("web_search", {})
        if not web_search_config.get("enabled", False):
            raise ValueError("Web search is disabled in configuration")

        provider_name = web_search_config.get("provider", "tavily").lower()

        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unsupported web search provider: {provider_name}. Available providers: {available}")

        provider_instance = cls._providers[provider_name]()
        logger.info(f"[WebSearchProviderFactory] Provider initialized: {provider_name}")
        return provider_instance

web_search_factory = WebSearchProviderFactory()
