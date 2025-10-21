from enum import Enum

class WorkflowNodeName(str, Enum):
    INTENT_ANALYSIS = "intent_analysis"
    SMART_ROUTER = "smart_router"
    WEB_SEARCH_AGENT = "web_search_agent"
    MULTI_TOOL_ORCHESTRATOR = "multi_tool_orchestrator"
    CONTEXT_AGGREGATION = "context_aggregation"
    RELEVANCE_RANKING = "relevance_ranking"
    CONTEXT_LIMIT_CHECK = "context_limit_check"
    PROMPT_CONSTRUCTION = "prompt_construction"
    LLM_RESPONSE_GENERATION = "llm_response_generation"
    RESPONSE_ENHANCEMENT = "response_enhancement"
    RESPONSE_VALIDATION = "response_validation"
    DEFAULT = "default"


