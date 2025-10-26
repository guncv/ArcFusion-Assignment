from enum import Enum

class LLMType(Enum):
    INIT_ROUTER_AGENT = "init_router_agent"
    CLARIFICATION_AGENT = "clarification_agent"
    SMALLTALK_AGENT = "smalltalk_agent"
    NEEDS_MORE_DETAIL_AGENT = "needs_more_detail_agent"
    REFINED_QUERY_AGENT = "refined_query_agent"
    
    # Orchestration agents
    PLANNER_AGENT = "planner_agent"
    TOOL_EXECUTOR = "tool_executor"
    WEB_SEARCH_AGENT = "web_search_agent"
    SYNTHESIZER_AGENT = "synthesizer_agent"
    REFLECTION_AGENT = "reflection_agent"
    RESPONSE_EVALUATOR = "response_evaluator"

class LLMAgentName(Enum):
    INIT_ROUTER_AGENT = "InitRouterAgent"
    CLARIFICATION_AGENT = "ClarificationAgent"
    SMALLTALK_AGENT = "SmallTalkAgent"
    NEEDS_MORE_DETAIL_AGENT = "NeedsMoreDetailAgent"
    REFINED_QUERY_AGENT = "RefinedQueryAgent"
    INTENT_ANALYSIS_AGENT = "IntentAnalysisAgent"
    PLANNER_AGENT = "PlannerAgent"
    TOOL_EXECUTOR = "ToolExecutor"
    WEB_SEARCH_AGENT = "WebSearchAgent"
    SYNTHESIZER_AGENT = "SynthesizerAgent"
    REFLECTION_AGENT = "ReflectionAgent"
