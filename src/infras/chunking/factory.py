from typing import Dict, Any, Optional
from functools import lru_cache

from src.config import config
from infras.rag.chunking.base import BaseChunker
from infras.rag.chunking import SemanticChunker
from infras.rag.chunking import FixedSizeChunker
from infras.embedding.factory import EmbeddingFactory


class ChunkerFactory:

    _chunkers = {
        "semantic": SemanticChunker,
        "semantic_llamaindex": SemanticChunker,
        "fixed_size": FixedSizeChunker,
    }

    @classmethod
    @lru_cache(maxsize=1)
    def get(cls, chunker_name: Optional[str] = None) -> BaseChunker:
        rag_config = config.get("rag", {})
        chunking_config = rag_config.get("chunking", {})

        if not chunking_config:
            raise ValueError("Chunking configuration not found")

        # Use provided strategy name or get from config
        strategy = strategy_name or chunking_config.get("strategy", "semantic_llamaindex")

        # Build strategy-specific config
        strategy_config = {}

        # Map config fields to strategy parameters
        if strategy in ["semantic", "semantic_llamaindex"]:
            strategy_config["buffer_size"] = chunking_config.get("llamaindex_buffer_size", 1)
            strategy_config["breakpoint_percentile_threshold"] = chunking_config.get(
                "llamaindex_breakpoint_threshold", 95
            )
        elif strategy == "fixed_size":
            strategy_config["chunk_size"] = chunking_config.get("chunk_size", 512)
            strategy_config["chunk_overlap"] = chunking_config.get("chunk_overlap", 50)

        # Get embedding provider if needed
        embedding_provider = None
        if strategy in ["semantic", "semantic_llamaindex"]:
            embedding_provider = EmbeddingFactory.get()

        return cls.create(strategy, strategy_config, embedding_provider)

    @classmethod
    def create(
        cls,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        embedding_provider = None
    ) -> BaseChunker:
        strategy_name = strategy_name.lower()

        if strategy_name not in cls._chunkers:
            available = ", ".join(cls._chunkers.keys())
            raise ValueError(
                f"Unsupported chunking strategy: {strategy_name}. "
                f"Available strategies: {available}"
            )

        strategy_class = cls._chunkers[strategy_name]

        # Some strategies require embedding provider
        if chunker_name in ["semantic", "semantic_llamaindex"]:
            if embedding_provider is None:
                raise ValueError(
                    f"Embedding provider is required for '{chunker_name}' chunking strategy"
                )
            return strategy_class(embedding_provider=embedding_provider, **config)
        else:
            return strategy_class(**config)

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        Register a new chunking strategy.

        Args:
            name: Strategy name
            strategy_class: Strategy class (must inherit from BaseChunker)
        """
        if not issubclass(strategy_class, BaseChunker):
            raise TypeError(
                f"Strategy class must inherit from BaseChunker"
            )
        cls._strategies[name.lower()] = strategy_class

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """
        Get list of available strategy names.

        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())
