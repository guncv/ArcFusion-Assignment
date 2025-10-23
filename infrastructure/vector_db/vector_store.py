from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from core.log.logger import logger
from core.config.config import nested_config as config


class VectorStoreManager:
    def __init__(self):
        self.collection_name = config["vector_db"]["collection_name"]
        self.embedding_model = config["vector_db"]["embedding_model"]
        self.embedder_api_key = config["vector_db"]["embedder_api_key"]
        self.persist_directory = config["vector_db"]["chroma"]["persist_directory"]

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.embedder_api_key
        )

        self.vectorstore: Optional[Chroma] = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        except Exception as e:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        if not documents:
            logger.warning("[VectorStoreManager] No documents to add")
            return []

        try:
            all_ids = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                ids = self.vectorstore.add_documents(batch)
                all_ids.extend(ids)

                logger.info(
                    f"[VectorStoreManager] Added batch {i // batch_size + 1}: "
                    f"{len(batch)} documents"
                )

            logger.info(
                f"[VectorStoreManager] Successfully added {len(documents)} documents "
                f"to collection '{self.collection_name}'"
            )

            return all_ids

        except Exception as e:
            logger.error(f"[VectorStoreManager] Error adding documents: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter
            )

            logger.info(
                f"[VectorStoreManager] Similarity search for '{query[:50]}...' "
                f"returned {len(results)} results"
            )

            return results

        except Exception as e:
            logger.error(f"[VectorStoreManager] Error in similarity search: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        try:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filter
            )

            logger.info(
                f"[VectorStoreManager] Search with scores for '{query[:50]}...' "
                f"returned {len(results)} results"
            )

            return results

        except Exception as e:
            logger.error(f"[VectorStoreManager] Error in search with scores: {e}")
            return []

    def as_retriever(self, **kwargs):
        return self.vectorstore.as_retriever(**kwargs)

    def delete_collection(self):
        try:
            self.vectorstore.delete_collection()
            logger.info(f"[VectorStoreManager] Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"[VectorStoreManager] Error deleting collection: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            collection = self.vectorstore._collection
            count = collection.count()

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model,
            }
        except Exception as e:
            logger.error(f"[VectorStoreManager] Error getting stats: {e}")
            return {}