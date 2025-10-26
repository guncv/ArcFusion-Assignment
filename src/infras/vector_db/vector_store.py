from typing import List, Optional
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from src.config import config

class VectorDBManager:
    def __init__(self):
        self.collection_name = config["vector_db"]["collection_name"]
        self.embedding_model = config["vector_db"]["embedding_model"]
        self.embedder_api_key = config["vector_db"]["embedder_api_key"]
        self.persist_directory = config["vector_db"]["chroma"]["persist_directory"]

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.embed_model = OpenAIEmbedding(
            model_name=self.embedding_model,
            api_key=self.embedder_api_key
        )

        self.index: Optional[VectorStoreIndex] = None
        self.chroma_collection = None
        
        chroma_client = chromadb.PersistentClient(path=self.persist_directory)

        self.chroma_collection = chroma_client.get_or_create_collection(
            name=self.collection_name
        )

        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self.embed_model
        )

    def add_documents(self, nodes: List[TextNode], batch_size: int = 100):
        if not nodes:
            return []

        try:
            all_ids = []

            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]

                self.index.insert_nodes(batch)

                batch_ids = [node.node_id for node in batch]
                all_ids.extend(batch_ids)
                
            return all_ids

        except Exception as e:
            raise

    def get_retriever(self, similarity_top_k: int = 3):

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
        )

        return retriever

vector_db_manager = VectorDBManager()