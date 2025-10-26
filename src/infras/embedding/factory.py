from typing import Dict, Any
from functools import lru_cache
from src.config import config
from infras.embedding.base import BaseEmbedding
from infras.embedding.providers.openai import OpenAIEmbeddingProvider

class EmbeddingFactory:
    _providers = {
        "openai": OpenAIEmbeddingProvider,
    }

    @classmethod
    @lru_cache(maxsize=1)
    def get(cls) -> BaseEmbedding:
        embedding_config = config.get("embedding", {})

        if not embedding_config:
            raise ValueError("Embedding configuration not found")

        provider = embedding_config.get("provider", "openai")
        provider_config = embedding_config.get(provider, {})

        if not provider_config or not provider:
            raise ValueError(f"No configuration found for embedding provider: {provider}")

        return cls.create(provider, provider_config)

    @classmethod
    def create(cls, provider_name: str, provider_config: Dict[str, Any]) -> BaseEmbedding:
        provider_name = provider_name.lower()

        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported embedding provider: {provider_name}. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class(**provider_config)