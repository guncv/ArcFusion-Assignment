from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document
from infrastructure.rag.document_processor import DocumentProcessor
from infrastructure.vector_db.vector_store import VectorStoreManager
from .retrievers import HybridRetrieverManager
from core.log.logger import logger
from core.config.config import nested_config as config


class RAGPipeline:
    def __init__(self):
        self.collection_name = config["vector_db"]["collection_name"]
        self.top_k = config["rag"]["retrieval"]["top_k"]

        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()

        self.hybrid_retriever: Optional[HybridRetrieverManager] = HybridRetrieverManager()

    def ingest_documents(self, pdf_paths: List[str], clear_existing: bool = False) -> Dict[str, Any]:
        try:
            if clear_existing:
                self.vector_store.delete_collection()
                self.vector_store = VectorStoreManager()

            chunks = self.doc_processor.process_multiple_pdfs(pdf_paths)

            if not chunks:
                return {"status": "failed", "reason": "No content extracted"}

            doc_ids = self.vector_store.add_documents(chunks)

            stats = {
                "status": "success",
                "pdfs_processed": len(pdf_paths),
                "chunks_created": len(chunks),
                "documents_indexed": len(doc_ids),
                "collection_name": self.collection_name,
            }

            logger.info(f"[RAGPipeline] Ingestion complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"[RAGPipeline] Error during ingestion: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def retrieve(self, query: str) -> Dict[str, Any]:
        try:
            local_docs = []
            local_docs = self.hybrid_retriever.retrieve(query)
            
            logger.info(f"[RAGPipeline] Retrieved documents: {local_docs}")

            return {
                "local_docs": local_docs
            }

        except Exception as e:
            logger.error(f"[RAGPipeline] Error during retrieval: {e}", exc_info=True)
            return {"local_docs": []}

    def _extract_sources(
        self,
        local_docs: List[Document]
    ) -> List[Dict[str, str]]:
        sources = []

        # Local sources
        for doc in local_docs:
            sources.append({
                "type": "document",
                "source": Path(doc.metadata.get("source", "Unknown")).name,
                "page": str(doc.metadata.get("page", "?")),
            })

        return sources

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.vector_store.get_collection_stats()
        stats["top_k"] = self.top_k
        return stats
