from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever,EnsembleRetriever
from core.log.logger import logger
from infrastructure.vector_db.vector_store import VectorStoreManager
from core.config.config import nested_config as config

class HybridRetrieverManager:

    def __init__(self, documents: Optional[List[Document]] = None):
        self.vector_store = VectorStoreManager()
        self.documents = documents or []
        self.vector_weight = config["rag"]["retrieval"]["vector_weight"]
        self.bm25_weight = config["rag"]["retrieval"]["bm25_weight"]
        self.k = config["rag"]["retrieval"]["top_k"]

        self.vector_retriever: Optional[BaseRetriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None

        self._initialize_retrievers()

    def _initialize_retrievers(self):
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )

        if self.documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)

            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[self.vector_weight, self.bm25_weight]
            )
        else:
            logger.warning(
                "[HybridRetriever] No documents provided for BM25. "
                "Falling back to vector-only retrieval."
            )
            self.ensemble_retriever = self.vector_retriever

    def retrieve(self, query: str) -> List[Document]:
        try:
            if self.ensemble_retriever:
                results = self.ensemble_retriever.get_relevant_documents(query)
            else:
                results = self.vector_retriever.get_relevant_documents(query)

            return results

        except Exception as e:
            logger.error(f"[HybridRetriever] Error during retrieval: {e}")
            return []

    def update_bm25_index(self, documents: List[Document]):
        self.documents = documents
        self._initialize_retrievers()
        logger.info(f"[HybridRetriever] Updated BM25 index with {len(documents)} documents")

    def get_retriever(self) -> BaseRetriever:
        if self.ensemble_retriever:
            return self.ensemble_retriever
        return self.vector_retriever