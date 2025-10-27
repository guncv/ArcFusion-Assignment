from typing import Dict, Any, Optional
from functools import lru_cache

from src.config import config
from src.infras.retrieval.base import BaseRetriever
from src.infras.retrieval.providers.vector import VectorRetriever
from src.infras.vector_db.factory import VectorStoreFactory
from src.infras.vector_db.base import BaseVectorStore

class RetrieverFactory:

    _retrievers = {
        "vector": VectorRetriever,
    }

    @classmethod
    @lru_cache(maxsize=1)
    def get(cls, retriever_type: Optional[str] = None) -> BaseRetriever:
        rag_config = config.get("rag", {})
        retrieval_config = rag_config.get("retrieval", {})

        if not retrieval_config:
            raise ValueError("Retrieval configuration not found")

        # Use provided type or default to "vector"
        ret_type = retriever_type or retrieval_config.get("type", "vector")

        # Get vector store
        vector_store = VectorStoreFactory.get()

        return cls.create(ret_type, vector_store)

    @classmethod
    def create(cls, retriever_type: str, vector_store: BaseVectorStore) -> BaseRetriever:
        retriever_type = retriever_type.lower()

        if retriever_type not in cls._retrievers:
            available = ", ".join(cls._retrievers.keys())
            raise ValueError(
                f"Unsupported retriever type: {retriever_type}. "
                f"Available retrievers: {available}"
            )

        retriever_class = cls._retrievers[retriever_type]

        # Pass vector store to the retriever
        return retriever_class(vector_store=vector_store)