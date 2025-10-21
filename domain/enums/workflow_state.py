from enum import Enum
from typing import TypedDict, Optional

class RoutingDecision(Enum):
    CLEAR_QUESTION = "clear_question"
    AMBIGUOUS = "ambiguous"
    
class WorkflowState(TypedDict, total=False):
    user_query: str
    routing_decision: str

    response: str
    error_message: Optional[str]