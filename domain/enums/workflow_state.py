from enum import Enum
from typing import TypedDict, Optional

class RoutingDecision(Enum):
    # Router Agent decisions
    CLEAR_QUESTION = "clear_question"
    AMBIGUOUS = "ambiguous"

    # Clarification Agent decisions
    SMALLTALK = "smalltalk"
    NEEDS_MORE_DETAIL = "needs_more_detail"
    PROCESS_QUERY = "process_query"

class WorkflowState(TypedDict, total=False):
    user_query: str
    session_id: str
    routing_decision: str
    refined_query: str

    # Planning fields
    execution_plan: str  # The plan description from planner
    planned_tools: list  # List of tools to execute ['rag_search', 'web_search', 'hybrid_search', 'rag_then_web', 'none']
    tool_results: dict   # Results from each tool execution
    previous_attempts: list  # History of previous planning attempts
    planning_attempts: int  # Number of times planner has been called

    # RAG-related fields
    retrieved_documents: list
    retrieval_scores: list
    confidence_score: float
    needs_web_search: bool
    web_search_results: list

    # Response fields
    response: str
    sources: list

    # RAG Reflection fields (first stage - evaluates RAG sufficiency)
    is_rag_sufficient: bool
    rag_quality_score: float
    rag_reflection_reasoning: str
    needs_orchestration: bool  # If RAG is insufficient, need web search

    # Orchestration Reflection fields (second stage - evaluates final answer)
    synthesis_attempts: int
    is_answer_sufficient: bool
    answer_quality_score: float
    reflection_issues: list
    reflection_suggestions: list

    error_message: Optional[str]