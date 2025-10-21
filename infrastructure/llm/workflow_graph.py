from typing import Literal
from langgraph.graph import StateGraph, END
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from agent.workflow_agents import (
    RouterAgent,
    ClarificationAgent,
    IntentAnalyzerAgent,
    RAGRetrievalAgent,
    WebSearchAgent,
    HybridAgent,
    VectorDBSearchAgent,
    ResponseGeneratorAgent,
    SessionMemoryAgent
)

class WorkflowGraph:
    """LangGraph workflow for the personal assistant"""

    def __init__(self):
        logger.info("Initializing WorkflowGraph")

        # Initialize all agents
        self.router_agent = RouterAgent()
        self.clarification_agent = ClarificationAgent()
        self.intent_analyzer_agent = IntentAnalyzerAgent()
        self.rag_retrieval_agent = RAGRetrievalAgent()
        self.web_search_agent = WebSearchAgent()
        self.hybrid_agent = HybridAgent(
            rag_agent=self.rag_retrieval_agent,
            web_search_agent=self.web_search_agent
        )
        self.vector_db_search_agent = VectorDBSearchAgent()
        self.response_generator_agent = ResponseGeneratorAgent()
        self.session_memory_agent = SessionMemoryAgent()

        # Build the graph
        self.graph = self._build_graph()
        logger.info("WorkflowGraph initialized successfully")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)

        # Add all nodes
        workflow.add_node("user_query", self._user_query_node)
        workflow.add_node("router_agent", self.router_agent)
        workflow.add_node("query_analysis", self._query_analysis_decision)
        workflow.add_node("clarification_agent", self.clarification_agent)
        workflow.add_node("refined_query", self._refined_query_node)
        workflow.add_node("intent_analyzer", self.intent_analyzer_agent)
        workflow.add_node("decision_node", self._decision_node)
        workflow.add_node("rag_retrieval_agent", self.rag_retrieval_agent)
        workflow.add_node("web_search_agent", self.web_search_agent)
        workflow.add_node("hybrid_agent", self.hybrid_agent)
        workflow.add_node("vector_db_search", self.vector_db_search_agent)
        workflow.add_node("web_search_api", self._web_search_api_node)
        workflow.add_node("response_generator", self.response_generator_agent)
        workflow.add_node("user_response", self._user_response_node)
        workflow.add_node("update_session_memory", self.session_memory_agent)
        workflow.add_node("return_to_user", self._return_to_user_node)

        # Set entry point
        workflow.set_entry_point("user_query")

        # Add edges from user_query
        workflow.add_edge("user_query", "router_agent")

        # Add edges from router_agent
        workflow.add_edge("router_agent", "query_analysis")

        # Add conditional edges from query_analysis
        workflow.add_conditional_edges(
            "query_analysis",
            self._route_query_analysis,
            {
                "ambiguous": "clarification_agent",
                "clear": "intent_analyzer"
            }
        )

        # Add edges from clarification_agent
        workflow.add_edge("clarification_agent", "refined_query")

        # Add edges from refined_query
        workflow.add_edge("refined_query", "intent_analyzer")

        # Add edges from intent_analyzer
        workflow.add_edge("intent_analyzer", "decision_node")

        # Add conditional edges from decision_node
        workflow.add_conditional_edges(
            "decision_node",
            self._route_decision,
            {
                "pdf_content": "rag_retrieval_agent",
                "external_info": "web_search_agent",
                "both": "hybrid_agent"
            }
        )

        # Add edges from RAG retrieval agent to vector DB search
        workflow.add_edge("rag_retrieval_agent", "vector_db_search")

        # Add edges from web search agent to web search API
        workflow.add_edge("web_search_agent", "web_search_api")

        # Add edges from hybrid agent to both vector DB and web search
        workflow.add_edge("hybrid_agent", "vector_db_search")
        workflow.add_edge("hybrid_agent", "web_search_api")

        # Add edges from vector_db_search to response_generator
        workflow.add_edge("vector_db_search", "response_generator")

        # Add edges from web_search_api to response_generator
        workflow.add_edge("web_search_api", "response_generator")

        # Add edges from response_generator
        workflow.add_edge("response_generator", "user_response")

        # Add edges from user_response
        workflow.add_edge("user_response", "update_session_memory")

        # Add edges from update_session_memory
        workflow.add_edge("update_session_memory", "return_to_user")

        # Set finish point
        workflow.add_edge("return_to_user", END)

        return workflow.compile()

    # Node functions
    def _user_query_node(self, state: WorkflowState) -> WorkflowState:
        """Entry point: User query"""
        logger.info("Node: User Query")
        return state

    def _query_analysis_decision(self, state: WorkflowState) -> WorkflowState:
        """Decision node for query analysis"""
        logger.info("Node: Query Analysis Decision")
        return state

    def _refined_query_node(self, state: WorkflowState) -> WorkflowState:
        """Node after clarification"""
        logger.info("Node: Refined Query")
        return state

    def _decision_node(self, state: WorkflowState) -> WorkflowState:
        """Decision node for retrieval strategy"""
        logger.info("Node: Decision Node")
        return state

    def _web_search_api_node(self, state: WorkflowState) -> WorkflowState:
        """Web Search API node"""
        logger.info("Node: Web Search API")
        return state

    def _user_response_node(self, state: WorkflowState) -> WorkflowState:
        """User response node"""
        logger.info("Node: User Response")
        return state

    def _return_to_user_node(self, state: WorkflowState) -> WorkflowState:
        """Final node before returning"""
        logger.info("Node: Return to User")
        return state

    # Routing functions
    def _route_query_analysis(self, state: WorkflowState) -> Literal["ambiguous", "clear"]:
        """Route based on query clarity"""
        query_clarity = state.get("query_clarity", "clear")
        logger.info(f"Routing query analysis: {query_clarity}")
        return "ambiguous" if query_clarity == "ambiguous" else "clear"

    def _route_decision(self, state: WorkflowState) -> Literal["pdf_content", "external_info", "both"]:
        """Route based on intent decision"""
        decision = state.get("decision", "both")
        logger.info(f"Routing decision: {decision}")
        return decision

    def invoke(self, user_input: str) -> WorkflowState:
        """Execute the workflow"""
        logger.info(f"Invoking workflow with input: {user_input}")

        initial_state: WorkflowState = {
            "user_query": user_input
        }

        try:
            result = self.graph.invoke(initial_state)
            logger.info("Workflow completed successfully")
            return result
        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            return {
                **initial_state,
                "error_message": str(e),
                "final_response": "An error occurred while processing your request."
            }

    async def ainvoke(self, user_input: str) -> WorkflowState:
        """Execute the workflow asynchronously"""
        logger.info(f"Async invoking workflow with input: {user_input}")

        initial_state: WorkflowState = {
            "user_query": user_input
        }

        try:
            result = await self.graph.ainvoke(initial_state)
            logger.info("Async workflow completed successfully")
            return result
        except Exception as e:
            logger.error(f"Async workflow error: {e}", exc_info=True)
            return {
                **initial_state,
                "error_message": str(e),
                "final_response": "An error occurred while processing your request."
            }


# Create singleton instance
workflow_graph = WorkflowGraph()