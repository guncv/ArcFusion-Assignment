from typing import List
from langchain_core.documents import Document as LangChainDocument
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from src.config import config
from src.infras.retrieval.base import BaseRetriever
from src.infras.vector_db.base import BaseVectorStore


class VectorRetriever(BaseRetriever):
    def __init__(self, vector_store: BaseVectorStore):
        vector_retrieval_config = config.get("rag", {}).get("retrieval", {})
        self._vector_store = vector_store
        self._similarity_top_k = vector_retrieval_config.get("similarity_top_k", 5)
        self._use_reranker = vector_retrieval_config.get("use_reranker", True)
        self._reranker_model = vector_retrieval_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._reranker_top_n = vector_retrieval_config.get("reranker_top_n", 3)

    def get_relevant_documents(self, query: str, **kwargs) -> List[LangChainDocument]:
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
        return "vector" if not self._use_reranker else "vector_with_reranking"
