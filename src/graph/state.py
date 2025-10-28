from enum import Enum
from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.documents import Document

class RoutingDecision(Enum):
    # Router Agent decisions
    CLEAR_QUESTION = "clear_question"
    AMBIGUOUS = "ambiguous"

    # Clarification Agent decisions
    SMALLTALK = "smalltalk"
    NEEDS_MORE_DETAIL = "needs_more_detail"
    PROCESS_QUERY = "process_query"

class ToolType(Enum):
    # Tool selection for planner
    RAG_SEARCH = "rag_search"  # Search internal documents/knowledge base
    WEB_SEARCH = "web_search"  # Search external web sources
    HYBRID_SEARCH = "hybrid_search"  # Search both RAG and Web in parallel

class ReflectionAction(Enum):
    # Reflection Agent next action decisions
    DONE = "done"  # Answer is complete, end workflow
    SWITCH_TO_WEBSEARCH = "switch_to_websearch"  # RAG failed, switch to web
    REPLAN_WITH_DIFFERENT_QUERY = "replan_with_different_query"  # Try different query (same tool)
    ADD_WEB_DETAILS = "add_web_details"  # Add more information via web search

class RetrievedDocument(TypedDict):
    document: Document
    score: float

class WorkflowState(TypedDict, total=False):
    user_query: str
    session_id: str
    routing_decision: str

    # Planning fields
    selected_tool: str  # Tool selected by planner: ToolType enum values
    generated_queries: str  # Single query for rag_search or web_search
    rag_query: str  # RAG-specific query for hybrid_search
    web_query: str  # Web-specific query for hybrid_search
    web_search_results: list  # Only populated for web_search tasks
    
    # RAG Retrieval fields
    retrieved_documents_with_scores: List[RetrievedDocument]
    
    # Meta-Assessor fields
    is_done: bool
    comment: str
    autonomous_attempts: int = 0

    # Response fields
    response: str  # Final merged response (current + old)
    error_message: Optional[str]

    # Evaluation metrics (for monitoring in production)
    factual_consistency: str  # "consistent" | "partial" | "unsupported" (for WebSearch)
    retrieval_quality: float  # 0-1 score (average cosine similarity)
    relevance_score: float  # 0-1 score (for web search relevance)
    confidence_score: float  # 0-1 combined confidence score