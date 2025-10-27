from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document


class BaseWebSearchProvider(ABC):

    @abstractmethod
    def search(self, query: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def search_as_documents(self, query: str) -> List[Document]:
        pass