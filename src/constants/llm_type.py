from enum import Enum

class LLMType(Enum):
    INIT_ROUTER_AGENT = "init_router_agent"
    CLARIFICATION_AGENT = "clarification_agent"
    SMALLTALK_AGENT = "smalltalk_agent"
    NEEDS_MORE_DETAIL_AGENT = "needs_more_detail_agent"
    REFINED_QUERY_AGENT = "refined_query_agent"
    
    # Orchestration agents
    PLANNER_AGENT = "planner_agent"
    RAG_RETRIEVAL_AGENT = "rag_retrieval_agent"
    WEB_SEARCH_AGENT = "web_search_agent"
    HYBRID_RETRIEVAL_AGENT = "hybrid_retrieval_agent"
    META_ASSESSOR_AGENT = "meta_assessor_agent"
    SYNTHESIZER_AGENT = "synthesizer_agent"
    REFLECTION_AGENT = "reflection_agent"
    EVALUATION_AGENT = "evaluation_agent"

class LLMAgentName(Enum):
    INIT_ROUTER_AGENT = "InitRouterAgent"
    CLARIFICATION_AGENT = "ClarificationAgent"
    SMALLTALK_AGENT = "SmallTalkAgent"
    NEEDS_MORE_DETAIL_AGENT = "NeedsMoreDetailAgent"
    REFINED_QUERY_AGENT = "RefinedQueryAgent"
    PLANNER_AGENT = "PlannerAgent"
    RAG_RETRIEVAL_AGENT = "RAGRetrievalAgent"
    WEB_SEARCH_AGENT = "WebSearchAgent"
    HYBRID_RETRIEVAL_AGENT = "HybridRetrievalAgent"
    META_ASSESSOR_AGENT = "MetaAssessorAgent"
    SYNTHESIZER_AGENT = "SynthesizerAgent"
    REFLECTION_AGENT = "ReflectionAgent"
    EVALUATION_AGENT = "EvaluationAgent"
