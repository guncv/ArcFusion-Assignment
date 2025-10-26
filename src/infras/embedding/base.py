from abc import ABC, abstractmethod
from typing import List, Any
from llama_index.core.base.embeddings.base import BaseEmbedding as LlamaBaseEmbedding


class BaseEmbedding(ABC):

    @abstractmethod
    def get_text_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def get_query_embedding(self, query: str) -> List[float]:
        pass
        
    @abstractmethod
    def get_text_embedding_model(self) -> LlamaBaseEmbedding:
        pass