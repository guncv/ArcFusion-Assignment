
from functools import lru_cache

from src.config import config
from src.infras.chunking.base import BaseChunker
from src.infras.chunking.provider.semantic_node import SemanticNodeChunker
from src.infras.embedding.factory import EmbeddingFactory

class ChunkerFactory:
    _providers = {
        "semantic_node": SemanticNodeChunker,
    }

    @classmethod
    @lru_cache(maxsize=1)
    def get(cls) -> BaseChunker:
        rag_config = config.get("rag", {})
        chunking_config = rag_config.get("chunking", {})

        if not chunking_config:
            raise ValueError("Chunking configuration not found in config")

        provider_type = chunking_config.get("provider", "semantic_node")

        return cls.create(provider_type=provider_type)

    @classmethod
    def create(cls, provider_type: str) -> BaseChunker:
        provider_type = provider_type.lower()

        if provider_type not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported chunking provider: {provider_type}. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_type]
        embedding_provider = EmbeddingFactory.get()

        return provider_class(embedding_provider=embedding_provider)