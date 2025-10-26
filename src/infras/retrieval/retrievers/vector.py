from typing import List, Optional
from langchain_core.documents import Document as LangChainDocument
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank

from infras.rag.retrieval.base import BaseRetriever
from infras.vector_db.base import BaseVectorStore


class VectorRetriever(BaseRetriever):
    """
    Vector-based retrieval with optional reranking.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        similarity_top_k: int = 5,
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_n: int = 3
    ):
        """
        Initialize vector retriever.

        Args:
            vector_store: Vector store instance
            similarity_top_k: Number of results to retrieve from vector store
            use_reranker: Whether to use reranking
            reranker_model: Reranker model name
            reranker_top_n: Number of results after reranking
        """
        self._vector_store = vector_store
        self._similarity_top_k = similarity_top_k
        self._use_reranker = use_reranker

        # Get base retriever from vector store
        self._base_retriever = vector_store.get_retriever(
            similarity_top_k=similarity_top_k
        )

        # Initialize reranker if enabled
        self._reranker = None
        if use_reranker:
            self._reranker = SentenceTransformerRerank(
                model=reranker_model,
                top_n=reranker_top_n
            )

    def get_relevant_documents(self, query: str, **kwargs) -> List[LangChainDocument]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            **kwargs: Additional parameters

        Returns:
            List of Document objects with relevance scores
        """
        try:
            # Retrieve from vector store
            nodes_with_scores = self._base_retriever.retrieve(query)

            # Apply reranking if enabled
            if self._reranker:
                query_bundle = QueryBundle(query_str=query)
                nodes_with_scores = self._reranker.postprocess_nodes(
                    nodes_with_scores,
                    query_bundle=query_bundle
                )

                # Update scores with rerank scores
                for node_with_score in nodes_with_scores:
                    rerank_score = float(node_with_score.score) if node_with_score.score is not None else 0.0
                    node_with_score.node.metadata["score"] = rerank_score
                    node_with_score.node.metadata["reranked"] = True

            # Convert to LangChain documents
            langchain_docs = self._llamaindex_to_langchain(nodes_with_scores)
            return langchain_docs

        except Exception as e:
            # Return empty list on error
            return []

    def _llamaindex_to_langchain(self, nodes: List[NodeWithScore]) -> List[LangChainDocument]:
        """
        Convert LlamaIndex nodes to LangChain documents.

        Args:
            nodes: List of NodeWithScore objects

        Returns:
            List of LangChain Document objects
        """
        documents = []
        for node_with_score in nodes:
            node = node_with_score.node
            score = float(node_with_score.score) if node_with_score.score is not None else 0.0

            doc = LangChainDocument(
                page_content=node.text if hasattr(node, 'text') else str(node),
                metadata={**node.metadata, "score": score}
            )
            documents.append(doc)
        return documents

    @property
    def retriever_type(self) -> str:
        """
        Get the type of retriever.

        Returns:
            Retriever type string
        """
        return "vector" if not self._use_reranker else "vector_with_reranking"
