from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval strategies.
    Allows easy switching between different retrieval approaches.
    """

    @abstractmethod
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            **kwargs: Additional retrieval-specific parameters

        Returns:
            List of Document objects with relevance scores
        """
        pass

    @property
    @abstractmethod
    def retriever_type(self) -> str:
        """
        Get the type of retriever.

        Returns:
            Retriever type string
        """
        pass
