from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from core.log.logger import logger
from infrastructure.vector_db.vector_store import VectorStoreManager
from core.config.config import nested_config as config

class SimpleEnsembleRetriever(BaseRetriever):
    """Simple ensemble retriever that combines multiple retrievers with weights."""

    retrievers: List[BaseRetriever]
    weights: List[float]

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from all retrievers and combine them."""
        all_docs = []
        seen_content = set()

        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for doc in docs:
                # Simple deduplication based on content
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    # Add weight to metadata
                    doc.metadata["relevance_score"] = doc.metadata.get("relevance_score", 0.5) * weight
                    all_docs.append(doc)

        # Sort by relevance score
        all_docs.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
        return all_docs



class HybridRetrieverManager:

    def __init__(self, documents: Optional[List[Document]] = None):
        self.vector_store = VectorStoreManager()
        self.documents = documents or []
        self.vector_weight = config["rag"]["retrieval"]["vector_weight"]
        self.bm25_weight = config["rag"]["retrieval"]["bm25_weight"]
        self.k = config["rag"]["retrieval"]["top_k"]

        self.vector_retriever: Optional[BaseRetriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[SimpleEnsembleRetriever] = None

        self._initialize_retrievers()

    def _initialize_retrievers(self):
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )

        if self.documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)

            self.ensemble_retriever = SimpleEnsembleRetriever(
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
                results = self.ensemble_retriever.invoke(query)
            else:
                results = self.vector_retriever.invoke(query)

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