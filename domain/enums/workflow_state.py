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

    response: str
    error_message: Optional[str]