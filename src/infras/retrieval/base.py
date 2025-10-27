from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseRetriever(ABC):

    @abstractmethod
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        pass

    @property
    @abstractmethod
    def retriever_type(self) -> str:
        pass
