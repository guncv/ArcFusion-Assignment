from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from core.log.logger import logger
from infrastructure.vector_db.vector_store import VectorStoreManager
from core.config.config import nested_config as config

class SimpleEnsembleRetriever(BaseRetriever):

    retrievers: List[BaseRetriever]
    weights: List[float]

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = []
        seen_content = set()

        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for doc in docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    doc.metadata["relevance_score"] = doc.metadata.get("relevance_score", 0.5) * weight
                    all_docs.append(doc)

        # Sort by relevance score
        all_docs.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
        return all_docs



class HybridRetrieverManager:

    def __init__(self):
        self.vector_store = VectorStoreManager()
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

        # Only create ensemble retriever if we have valid retrievers
        retrievers = [self.vector_retriever]
        weights = [self.vector_weight]
        
        if self.bm25_retriever is not None:
            retrievers.append(self.bm25_retriever)
            weights.append(self.bm25_weight)
        
        if len(retrievers) > 1:
            self.ensemble_retriever = SimpleEnsembleRetriever(
                retrievers=retrievers,
                weights=weights
            )
        else:
            # If only one retriever, don't create ensemble
            self.ensemble_retriever = None

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
        """Initialize BM25 retriever with documents and recreate ensemble retriever."""
        if documents:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = self.k
            logger.info(f"[HybridRetriever] Initialized BM25 retriever with {len(documents)} documents")
        else:
            self.bm25_retriever = None
            logger.warning("[HybridRetriever] No documents provided for BM25 initialization")
        
        # Reinitialize retrievers with updated BM25
        self._initialize_retrievers()

    def get_retriever(self) -> BaseRetriever:
        if self.ensemble_retriever:
            return self.ensemble_retriever
        return self.vector_retriever