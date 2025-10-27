from functools import lru_cache
from src.config import config
from src.infras.vector_db.base import BaseVectorStore
from src.infras.vector_db.providers.chroma import ChromaVectorStoreProvider
from src.infras.embedding.factory import EmbeddingFactory

class VectorStoreFactory:

    _providers = {
        "chroma": ChromaVectorStoreProvider,
    }

    @classmethod
    @lru_cache(maxsize=1)
    def get(cls) -> BaseVectorStore:
        vector_db_config = config.get("vector_db", {})

        if not vector_db_config:
            raise ValueError("Vector DB configuration not found")

        provider_name = vector_db_config.get("provider", "chroma")

        return cls.create(provider_name)

    @classmethod
    def create(cls, provider_name: str) -> BaseVectorStore:
        provider_name = provider_name.lower()

        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported vector store provider: {provider_name}. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_name]
        embedding_provider = EmbeddingFactory.get()

        return provider_class(embedding_provider)