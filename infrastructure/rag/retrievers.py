from typing import List
from langchain_core.documents import Document as LangChainDocument
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from infrastructure.vector_db.vector_store import VectorStoreManager

class RetrieverManager():
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        similarity_top_k: int = 5,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_n: int = 3
    ):
        self._base_retriever = vector_store_manager.get_retriever(similarity_top_k=similarity_top_k) 
        self._reranker = SentenceTransformerRerank(model=reranker_model, top_n=reranker_top_n)
        
    def _llamaindex_to_langchain(self, nodes: List[NodeWithScore]) -> List[LangChainDocument]:
        documents = []
        for node_with_score in nodes:
            node = node_with_score.node
            
            score = float(node_with_score.score) if node_with_score.score is not None else 0.0
            doc = LangChainDocument(
                page_content=node.text if hasattr(node, 'text') else str(node),
                metadata={"score": score}
            )
            documents.append(doc)
        return documents

    def get_relevant_documents(self, query: str) -> List[LangChainDocument]:
        try:
            nodes_with_scores = self._base_retriever.retrieve(query)

            if self._reranker:
                query_bundle = QueryBundle(query_str=query)
                nodes_with_scores = self._reranker.postprocess_nodes(
                    nodes_with_scores,
                    query_bundle=query_bundle
                )

                for node_with_score in nodes_with_scores:
                    # Convert numpy float32 to Python float for JSON serialization
                    rerank_score = float(node_with_score.score) if node_with_score.score is not None else 0.0
                    node_with_score.node.metadata["score"] = rerank_score

            langchain_docs = self._llamaindex_to_langchain(nodes_with_scores)
            return langchain_docs

        except Exception as e:
            return []
