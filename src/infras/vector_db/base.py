from abc import ABC, abstractmethod
from typing import List, Any, Optional
from llama_index.core.schema import TextNode
from langchain_core.documents import Document


class BaseVectorStore(ABC):

    @abstractmethod
    def add_documents(self, nodes: List[TextNode], batch_size: int = 100) -> List[str]:
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Document]:
        pass

    @abstractmethod
    def get_retriever(self, similarity_top_k: int = 5, **kwargs) -> Any:
        pass
