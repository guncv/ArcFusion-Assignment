from typing import List, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_core.documents import Document
import chromadb
from src.infras.vector_db.base import BaseVectorStore
from src.infras.embedding.base import BaseEmbedding
from src.config import config

class ChromaVectorStoreProvider(BaseVectorStore):

    def __init__(self, embedding_provider: BaseEmbedding):
        chroma_config = config.get("vector_db", {}).get("chroma", {})
        self._embedding_provider = embedding_provider
        self._collection_name = chroma_config.get("collection_name", "arcfusion_docs")
        self._persist_directory = chroma_config.get("persist_directory", "./.chroma")

        chroma_client = chromadb.PersistentClient(path=self._persist_directory)
        self._chroma_collection = chroma_client.get_or_create_collection(
            name=self._collection_name
        )

        vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embed_model = self._embedding_provider.get_text_embedding_model()

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
            raise Exception(f"Failed to add documents to Chroma: {str(e)}")

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
                    metadata={"score": score}
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