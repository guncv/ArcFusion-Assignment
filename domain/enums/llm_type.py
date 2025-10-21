from enum import Enum

class LLMType(Enum):
    ROUTER_AGENT = "router_agent"
    CLARIFICATION_AGENT = "clarification_agent"
    INTENT_ANALYSIS_AGENT = "intent_analysis_agent"
    RAG_RETRIEVAL_AGENT = "rag_retrieval_agent"
    WEB_SEARCH_AGENT = "web_search_agent"
    HYBRID_AGENT = "hybrid_agent"
    VECTOR_DB_SEARCH_AGENT = "vector_db_search_agent"
    RESPONSE_GENERATION_AGENT = "response_generation_agent"
    SESSION_MEMORY_AGENT = "session_memory_agent"
    