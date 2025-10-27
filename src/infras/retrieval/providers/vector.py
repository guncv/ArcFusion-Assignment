from typing import List
from langchain_core.documents import Document
from src.config import config
from src.infras.retrieval.base import BaseRetriever
from src.infras.vector_db.base import BaseVectorStore
class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: BaseVectorStore,
    ):
        vector_retrieval_config = config.get("rag", {}).get("retrieval", {})
        self._vector_store = vector_store
        self._similarity_top_k = vector_retrieval_config.get("similarity_top_k", 5)

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        try:
            documents = self._vector_store.similarity_search(query, k=self._similarity_top_k)
            return documents
        except Exception as e:
            raise ValueError(f"Failed to get relevant documents: {str(e)}")

    @property
    def retriever_type(self) -> str:
        return "vector"
