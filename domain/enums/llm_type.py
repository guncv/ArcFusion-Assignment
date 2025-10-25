from enum import Enum

class LLMType(Enum):
    INIT_ROUTER_AGENT = "init_router_agent"
    CLARIFICATION_AGENT = "clarification_agent"
    SMALLTALK_AGENT = "smalltalk_agent"
    NEEDS_MORE_DETAIL_AGENT = "needs_more_detail_agent"
    REFINED_QUERY_AGENT = "refined_query_agent"
    
    INTENT_ANALYSIS_AGENT = "intent_analysis_agent"
    
    # Orchestration agents
    PLANNER_AGENT = "planner_agent"
    TOOL_EXECUTOR = "tool_executor"
    WEB_SEARCH_AGENT = "web_search_agent"
    ORCHESTRATION_SYNTHESIZER_AGENT = "orchestration_synthesizer_agent"
    ORCHESTRATION_REFLECTION_AGENT = "orchestration_reflection_agent"
