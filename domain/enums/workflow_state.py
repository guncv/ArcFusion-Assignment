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

    # RAG-related fields
    retrieved_documents: list
    retrieval_scores: list
    confidence_score: float
    needs_web_search: bool
    web_search_results: list

    # Response fields
    response: str
    sources: list

    # Reflection fields
    synthesis_attempts: int
    is_answer_sufficient: bool
    answer_quality_score: float
    reflection_issues: list
    reflection_suggestions: list

    error_message: Optional[str]