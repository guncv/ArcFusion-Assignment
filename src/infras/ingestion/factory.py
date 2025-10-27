from functools import lru_cache
from src.config import config
from src.infras.ingestion.base import BaseReader
from src.infras.ingestion.providers.unstructured import UnstructuredReaderProvider


class ReaderFactory:
    # Registry of available reader providers
    _providers = {
        "unstructured": UnstructuredReaderProvider,
    }

    @classmethod
    @lru_cache(maxsize=1)
    def get(cls) -> BaseReader:
        rag_config = config.get("rag", {})
        ingestion_config = rag_config.get("ingestion", {})

        if not ingestion_config:
            # Default to unstructured if no config
            return cls.create(provider_type="unstructured")

        # Determine provider type (default: unstructured)
        provider_type = ingestion_config.get("provider", "unstructured")

        return cls.create(provider_type=provider_type)

    @classmethod
    def create(cls, provider_type: str = "unstructured") -> BaseReader:
        provider_type = provider_type.lower()

        if provider_type not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported reader provider: {provider_type}. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_type]

        return provider_class()