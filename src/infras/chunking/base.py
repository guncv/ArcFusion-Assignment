from abc import ABC, abstractmethod
from typing import List
from llama_index.core.schema import Document as LlamaDocument, TextNode


class BaseChunker(ABC):

    @abstractmethod
    def split_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        pass

    @abstractmethod
    def split_text(self, text: str, metadata: dict = None) -> List[TextNode]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
