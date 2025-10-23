"""RAG API Endpoints."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import os
import tempfile
from pathlib import Path

from domain.models.rag import (
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGStatsResponse,
    DocumentSource,
)
from infrastructure.rag import RAGPipeline
from core.log.logger import logger
from core.constants.constants import session_id_key

router = APIRouter()

# Global RAG pipeline instance
_rag_pipeline: RAGPipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            collection_name="arcfusion_docs",
            use_web_search=True,
            top_k=5
        )
    return _rag_pipeline


@router.post("/ingest", response_model=IngestDocumentsResponse)
async def ingest_documents(request: IngestDocumentsRequest):
    """
    Ingest PDF documents into the RAG system.

    Process PDFs, chunk them, and index in vector store.
    """
    try:
        logger.info(f"[RAG API] Ingesting {len(request.pdf_paths)} PDFs")

        pipeline = get_rag_pipeline()

        # Override collection if specified
        if request.collection_name != pipeline.collection_name:
            from infrastructure.rag.vector_store import VectorStoreManager
            pipeline.vector_store = VectorStoreManager(collection_name=request.collection_name)
            pipeline.collection_name = request.collection_name

        # Ingest documents
        result = pipeline.ingest_documents(
            pdf_paths=request.pdf_paths,
            clear_existing=request.clear_existing
        )

        return IngestDocumentsResponse(**result)

    except Exception as e:
        logger.error(f"[RAG API] Error ingesting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/upload", response_model=IngestDocumentsResponse)
async def upload_and_ingest(
    files: List[UploadFile] = File(...),
    clear_existing: bool = False
):
    """
    Upload PDF files and ingest into RAG system.

    Accepts multiple PDF files via multipart/form-data.
    """
    try:
        logger.info(f"[RAG API] Uploading and ingesting {len(files)} files")

        # Save uploaded files to temp directory
        temp_dir = tempfile.mkdtemp()
        pdf_paths = []

        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)

            pdf_paths.append(file_path)

        # Ingest documents
        pipeline = get_rag_pipeline()
        result = pipeline.ingest_documents(
            pdf_paths=pdf_paths,
            clear_existing=clear_existing
        )

        # Cleanup temp files
        for path in pdf_paths:
            try:
                os.remove(path)
            except:
                pass

        return IngestDocumentsResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[RAG API] Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Query the RAG system.

    Retrieves relevant documents and generates answer with sources.
    """
    try:
        logger.info(f"[RAG API] Query: '{request.query[:100]}...'")

        pipeline = get_rag_pipeline()

        # Generate answer
        result = await pipeline.generate(
            query=request.query,
            session_id=session_id_key,
            use_web_search=request.use_web_search,
            use_ensemble=request.use_ensemble
        )

        # Convert sources to DocumentSource models
        sources = [DocumentSource(**source) for source in result.get("sources", [])]

        return RAGQueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            num_sources=len(sources),
            used_web_search=result.get("used_web_search", False),
            query=request.query
        )

    except Exception as e:
        logger.error(f"[RAG API] Error querying RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """
    Get RAG system statistics.

    Returns info about vector store, collection, etc.
    """
    try:
        pipeline = get_rag_pipeline()
        stats = pipeline.get_stats()

        return RAGStatsResponse(**stats)

    except Exception as e:
        logger.error(f"[RAG API] Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collection/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a vector store collection.

    WARNING: This permanently deletes all documents in the collection.
    """
    try:
        from infrastructure.rag.vector_store import VectorStoreManager

        vector_store = VectorStoreManager(collection_name=collection_name)
        vector_store.delete_collection()

        return {"status": "success", "message": f"Deleted collection: {collection_name}"}

    except Exception as e:
        logger.error(f"[RAG API] Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
