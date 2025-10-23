from .document_processor import DocumentProcessor
from .web_search import TavilyWebSearch
from .rag_pipeline import RAGPipeline

# Lazy singleton instances
_rag_pipeline_instance = None
_tavily_web_search_instance = None
_document_processor_instance = None

def get_rag_pipeline() -> RAGPipeline:
    """Get the RAGPipeline singleton instance."""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance

def get_tavily_web_search() -> TavilyWebSearch:
    """Get the TavilyWebSearch singleton instance."""
    global _tavily_web_search_instance
    if _tavily_web_search_instance is None:
        _tavily_web_search_instance = TavilyWebSearch()
    return _tavily_web_search_instance

def get_document_processor() -> DocumentProcessor:
    """Get the DocumentProcessor singleton instance."""
    global _document_processor_instance
    if _document_processor_instance is None:
        _document_processor_instance = DocumentProcessor()
    return _document_processor_instance

__all__ = [
    "RAGPipeline",
    "TavilyWebSearch",
    "DocumentProcessor",
    "get_rag_pipeline",
    "get_tavily_web_search",
    "get_document_processor",
]
