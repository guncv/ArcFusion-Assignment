"""RAG API Endpoints."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import os
import tempfile
from pathlib import Path
from infrastructure.rag import get_rag_pipeline

from domain.models.rag import (
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGStatsResponse,
    DocumentSource,
)
from infrastructure.rag.auto_ingestion import perform_auto_ingestion, get_auto_ingestion_status
from core.log.logger import logger
from core.constants.constants import session_id_key

router = APIRouter()

@router.post("/ingest", response_model=IngestDocumentsResponse)
async def ingest_documents(request: IngestDocumentsRequest):
    try:
        logger.info(f"[RAG API] Ingesting {len(request.pdf_paths)} PDFs")

        pipeline = get_rag_pipeline()
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
    try:
        logger.info(f"[RAG API] Uploading and ingesting {len(files)} files")

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

        pipeline = get_rag_pipeline()
        result = pipeline.ingest_documents(
            pdf_paths=pdf_paths,
            clear_existing=clear_existing
        )

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
    try:
        logger.info(f"[RAG API] Query: '{request.query[:100]}...'")

        pipeline = get_rag_pipeline()

        result = await pipeline.generate(
            query=request.query,
            session_id=session_id_key,
        )

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


@router.post("/retrieve/vector-db")
async def retrieve_from_vector_db(request: RAGQueryRequest):
    """Retrieve documents from vector database only (no web search)."""
    try:
        logger.info(f"[RAG API] Vector DB retrieval: '{request.query[:100]}...'")

        pipeline = get_rag_pipeline()
        result = pipeline.retrieve(query=request.query, use_web_search=False)

        # Format local documents for response
        local_docs = []
        for doc in result.get("local_docs", []):
            local_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "vector_db"
            })

        return {
            "query": request.query,
            "documents": local_docs,
            "num_documents": len(local_docs),
            "source": "vector_db"
        }

    except Exception as e:
        logger.error(f"[RAG API] Error during vector DB retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve/web-search")
async def retrieve_from_web_search(request: RAGQueryRequest):
    """Retrieve documents from web search only (no vector DB)."""
    try:
        logger.info(f"[RAG API] Web search retrieval: '{request.query[:100]}...'")

        pipeline = get_rag_pipeline()
        
        # Only get web search results
        web_docs = []
        if pipeline.web_search:
            web_docs = pipeline.web_search.search_as_documents(request.query)

        # Format web documents for response
        formatted_docs = []
        for doc in web_docs:
            formatted_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "web_search"
            })

        return {
            "query": request.query,
            "documents": formatted_docs,
            "num_documents": len(formatted_docs),
            "source": "web_search"
        }

    except Exception as e:
        logger.error(f"[RAG API] Error during web search retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve/hybrid")
async def retrieve_hybrid(request: RAGQueryRequest):
    """Retrieve documents from both vector DB and web search."""
    try:
        logger.info(f"[RAG API] Hybrid retrieval: '{request.query[:100]}...'")

        pipeline = get_rag_pipeline()
        result = pipeline.retrieve(query=request.query)

        # Format documents for response
        local_docs = []
        for doc in result.get("local_docs", []):
            local_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "vector_db"
            })

        web_docs = []
        for doc in result.get("web_docs", []):
            web_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "web_search"
            })

        return {
            "query": request.query,
            "vector_db_documents": local_docs,
            "web_search_documents": web_docs,
            "num_vector_db_documents": len(local_docs),
            "num_web_search_documents": len(web_docs),
            "total_documents": len(local_docs) + len(web_docs)
        }

    except Exception as e:
        logger.error(f"[RAG API] Error during hybrid retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """Get RAG system statistics."""
    try:
        pipeline = get_rag_pipeline()
        
        stats = {
            "collection_name": pipeline.collection_name,
            "document_count": pipeline.vector_store.get_document_count(),
            "web_search_enabled": pipeline.use_web_search_flag,
            "top_k": pipeline.top_k,
        }
        
        return RAGStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"[RAG API] Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auto-ingestion/status")
async def get_auto_ingestion_status_endpoint():
    """Get auto-ingestion configuration and status."""
    try:
        status = get_auto_ingestion_status()
        return status
        
    except Exception as e:
        logger.error(f"[RAG API] Error getting auto-ingestion status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-ingestion/trigger")
async def trigger_auto_ingestion():
    """Manually trigger auto-ingestion."""
    try:
        logger.info("[RAG API] Manual auto-ingestion triggered")
        
        result = await perform_auto_ingestion()
        
        if result:
            if result.get("status") == "success":
                return {
                    "status": "success",
                    "message": "Auto-ingestion completed successfully",
                    "details": result
                }
            else:
                return {
                    "status": "failed",
                    "message": "Auto-ingestion failed",
                    "details": result
                }
        else:
            return {
                "status": "skipped",
                "message": "Auto-ingestion skipped (not configured or no documents found)"
            }
        
    except Exception as e:
        logger.error(f"[RAG API] Error triggering auto-ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))