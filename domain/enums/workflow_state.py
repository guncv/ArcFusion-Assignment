from typing import TypedDict, Optional, List, Dict, Any

class WorkflowState(TypedDict, total=False):
    # User Query
    user_query: str
    original_query: str
    refined_query: str

    # Query Analysis
    query_clarity: str  # "clear" or "ambiguous"
    clarification_needed: bool

    # Intent Analysis
    intent: str
    decision: str  # "pdf_content", "external_info", "both"

    # Retrieved Information
    rag_results: Optional[List[Dict[str, Any]]]
    web_search_results: Optional[List[Dict[str, Any]]]
    vector_db_results: Optional[List[Dict[str, Any]]]

    # Response Generation
    response: str
    final_response: str

    # Session & Error
    session_id: Optional[str]
    conversation_history: Optional[List[Dict[str, str]]]
    error_message: Optional[str]
    message: Optional[str]