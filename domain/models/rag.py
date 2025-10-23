"""Domain models for RAG system."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DocumentSource(BaseModel):
    """Source metadata for retrieved documents."""
    type: str = Field(description="Source type: 'document' or 'web'")
    source: Optional[str] = Field(None, description="Document filename or web URL")
    page: Optional[str] = Field(None, description="Page number for documents")
    title: Optional[str] = Field(None, description="Title for web sources")
    url: Optional[str] = Field(None, description="URL for web sources")


class IngestDocumentsRequest(BaseModel):
    """Request to ingest PDF documents."""
    pdf_paths: List[str] = Field(description="List of paths to PDF files")
    clear_existing: bool = Field(default=False, description="Clear existing collection before ingestion")
    collection_name: Optional[str] = Field(default="arcfusion_docs", description="Collection name")


class IngestDocumentsResponse(BaseModel):
    """Response from document ingestion."""
    status: str = Field(description="Status: 'success' or 'failed'")
    pdfs_processed: Optional[int] = Field(None, description="Number of PDFs processed")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created")
    documents_indexed: Optional[int] = Field(None, description="Number of documents indexed")
    collection_name: Optional[str] = Field(None, description="Collection name")
    error: Optional[str] = Field(None, description="Error message if failed")


class RAGQueryRequest(BaseModel):
    """Request for RAG query."""
    query: str = Field(description="User query")
    use_web_search: Optional[bool] = Field(None, description="Override web search setting")
    use_ensemble: bool = Field(default=True, description="Use ensemble retriever")
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve")


class RAGQueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str = Field(description="Generated answer")
    sources: List[DocumentSource] = Field(description="Source documents")
    num_sources: int = Field(description="Number of sources used")
    used_web_search: bool = Field(description="Whether web search was used")
    query: str = Field(description="Original query")


class RAGStatsResponse(BaseModel):
    """RAG system statistics."""
    collection_name: str = Field(description="Vector store collection name")
    document_count: int = Field(description="Number of documents in collection")
    persist_directory: str = Field(description="Persistence directory")
    embedding_model: str = Field(description="Embedding model used")
    web_search_enabled: bool = Field(description="Web search enabled")
    top_k: int = Field(description="Default top-k for retrieval")
