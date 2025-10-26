from typing import Dict, Any, Optional
from functools import lru_cache

from config import nested_config as config
from infras.rag.retrieval.base import BaseRetriever
from infras.rag.retrieval.retrievers.vector import VectorRetriever
from infras.vector_db.factory import VectorStoreFactory


class RetrieverFactory:
    """
    Factory class for creating retriever instances.
    Makes it easy to switch between different retrieval strategies via configuration.
    """

    _retrievers = {
        "vector": VectorRetriever,
        # Add more retrievers here as they're implemented:
        # "bm25": BM25Retriever,
        # "ensemble": EnsembleRetriever,
        # "hybrid": HybridRetriever,
    }

    @classmethod
    @lru_cache(maxsize=1)
    def get(cls, retriever_type: Optional[str] = None) -> BaseRetriever:
        """
        Get retriever instance (singleton via cache).

        Args:
            retriever_type: Optional retriever type override. If not provided, uses config.

        Returns:
            Retriever instance

        Raises:
            ValueError: If retrieval is not configured
        """
        rag_config = config.get("rag", {})
        retrieval_config = rag_config.get("retrieval", {})

        if not retrieval_config:
            raise ValueError("Retrieval configuration not found")

        # Use provided type or default to "vector"
        ret_type = retriever_type or retrieval_config.get("type", "vector")

        # Build retriever config
        retriever_config = {
            "similarity_top_k": retrieval_config.get("top_k", 5),
            "use_reranker": retrieval_config.get("use_reranker", True),
            "reranker_model": retrieval_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            "reranker_top_n": retrieval_config.get("reranker_top_n", 3),
        }

        # Get vector store
        vector_store = VectorStoreFactory.get()

        return cls.create(ret_type, retriever_config, vector_store)

    @classmethod
    def create(
        cls,
        retriever_type: str,
        retriever_config: Dict[str, Any],
        vector_store
    ) -> BaseRetriever:
        """
        Create a retriever instance based on the retriever type.

        Args:
            retriever_type: Type of retriever (e.g., "vector", "ensemble")
            config: Configuration dictionary for the retriever
            vector_store: Vector store instance to use

        Returns:
            An instance of the requested retriever

        Raises:
            ValueError: If the retriever type is not supported
        """
        retriever_type = retriever_type.lower()

        if retriever_type not in cls._retrievers:
            available = ", ".join(cls._retrievers.keys())
            raise ValueError(
                f"Unsupported retriever type: {retriever_type}. "
                f"Available retrievers: {available}"
            )

        retriever_class = cls._retrievers[retriever_type]

        # Pass vector store to the retriever
        return retriever_class(vector_store=vector_store, **config)

    @classmethod
    def register_retriever(cls, name: str, retriever_class: type):
        """
        Register a new retriever type.

        Args:
            name: Retriever name
            retriever_class: Retriever class (must inherit from BaseRetriever)
        """
        if not issubclass(retriever_class, BaseRetriever):
            raise TypeError(
                f"Retriever class must inherit from BaseRetriever"
            )
        cls._retrievers[name.lower()] = retriever_class

    @classmethod
    def get_available_retrievers(cls) -> list[str]:
        """
        Get list of available retriever types.

        Returns:
            List of retriever type names
        """
        return list(cls._retrievers.keys())
