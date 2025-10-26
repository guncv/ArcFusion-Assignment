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

    # Intent Analysis Agent decisions
    USE_RAG = "use_rag"  # Question can be answered from documents
    USE_WEB_SEARCH = "use_web_search"  # Question needs real-time/web data

    # RAG Reflection Agent decisions
    RAG_SUFFICIENT = "rag_sufficient"
    RAG_INSUFFICIENT = "rag_insufficient"
    NOT_RELEVANT = "not_relevant"

class ToolType(Enum):
    # Tool selection for planner
    RAG_SEARCH = "rag_search"  # Search internal documents/knowledge base
    WEB_SEARCH = "web_search"  # Search external web sources

class RetrievedDocument(TypedDict):
    document: Document
    score: float

class WorkflowState(TypedDict, total=False):
    user_query: str
    session_id: str
    routing_decision: str
    
    # RAG-related fields
    retrieved_documents_with_scores: List[RetrievedDocument]
    
    # RAG Synthesizer fields
    rag_synthesizer_response: str = ""
    
    # RAG Reflection fields
    rag_reflection_comment: str

    # Planning fields
    selected_tool: str  # Tool selected by planner: ToolType enum values
    generated_queries: list
    web_search_results: list  # Only populated for web_search tasks

    # Orchestration Reflection fields
    is_answer_sufficient: bool = False
    reflection_issues: str
    orchestration_attempts: int = 0
    old_queries: list  # Previous queries from previous orchestration attempts

    # Feedback loop history tracking (for debugging and analysis)
    orchestration_history: List[Dict[str, Any]]  # List of iteration attempts with details

    # Response fields
    current_synthesized_response: str  # Response from current iteration only (before merging with old)
    response: str  # Final merged response (current + old)
    error_message: Optional[str]

    # Evaluation metrics (for monitoring in production)
    evaluation_metrics: Dict[str, Any]  # Contains all evaluation results
    faithfulness: str  # "supported" | "partial" | "unsupported" (for RAG)
    factual_consistency: str  # "consistent" | "partial" | "unsupported" (for WebSearch)
    retrieval_quality: float  # 0-1 score (average cosine similarity)
    relevance_score: float  # 0-1 score (for web search relevance)
    confidence_score: float  # 0-1 combined confidence score