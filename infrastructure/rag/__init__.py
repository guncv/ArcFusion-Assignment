from .document_processor import DocumentProcessor
from .web_search import TavilyWebSearch
from .auto_ingestion import AutoIngestionManager

# Lazy singleton instances
_tavily_web_search_instance = None
_document_processor_instance = None
_auto_ingestion_manager_instance = None

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
    "TavilyWebSearch",
    "DocumentProcessor",
    "AutoIngestionManager",
    "get_tavily_web_search",
    "get_document_processor",
    "get_auto_ingestion_manager",
]
