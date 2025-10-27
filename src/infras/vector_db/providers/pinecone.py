from typing import List, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from src.infras.vector_db.base import BaseVectorStore
from src.infras.embedding.base import BaseEmbedding
from src.config import config

class PineconeVectorStoreProvider(BaseVectorStore):

    def __init__(self, embedding_provider: BaseEmbedding):
        pinecone_config = config.get("vector_db", {}).get("pinecone", {})
        self._embedding_provider = embedding_provider
        self._index_name = pinecone_config.get("index_name", "arcfusion_docs")
        self._dimensions = pinecone_config.get("dimensions", 1536)
        self._metric = pinecone_config.get("metric", "cosine")
        self._cloud = pinecone_config.get("cloud", "aws")
        self._region = pinecone_config.get("region", "us-east-1")
        self._api_key = pinecone_config.get("api_key")

        pc = Pinecone(api_key=self._api_key)

        if self._index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self._index_name,
                dimension=self._dimensions,
                metric=self._metric,
                spec=ServerlessSpec(
                    cloud=self._cloud,
                    region=self._region
                )
            )

        # Get the index
        self._pinecone_index = pc.Index(self._index_name)

        # Create vector store
        vector_store = PineconeVectorStore(
            pinecone_index=self._pinecone_index
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Get the underlying LlamaIndex embed model
        embed_model = self._embedding_provider.get_text_embedding_model()

        # Initialize index
        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=embed_model
        )

    def add_documents(self, nodes: List[TextNode], batch_size: int = 100) -> List[str]:
        if not nodes:
            return []

        try:
            all_ids = []

            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                self._index.insert_nodes(batch)
                batch_ids = [node.node_id for node in batch]
                all_ids.extend(batch_ids)

            return all_ids

        except Exception as e:
            raise Exception(f"Failed to add documents to Pinecone: {str(e)}")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Document]:
        try:
            retriever = self.get_retriever(similarity_top_k=k)
            nodes_with_scores: List[NodeWithScore] = retriever.retrieve(query)

            documents = []
            for node_with_score in nodes_with_scores:
                node = node_with_score.node
                score = float(node_with_score.score) if node_with_score.score is not None else 0.0

                doc = Document(
                    page_content=node.text if hasattr(node, 'text') else str(node),
                    metadata={
                        **node.metadata,
                        "score": score,
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            raise Exception(f"Failed to perform similarity search: {str(e)}")

    def get_retriever(self, similarity_top_k: int = 5, **kwargs) -> Any:
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=similarity_top_k,
            **kwargs
        )
        return retriever