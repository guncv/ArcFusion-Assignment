from abc import ABC, abstractmethod
from typing import List
from llama_index.core.base.embeddings.base import BaseEmbedding

class BaseEmbedding(ABC):

    @abstractmethod
    def get_text_embedding_model(self) -> BaseEmbedding:
        pass