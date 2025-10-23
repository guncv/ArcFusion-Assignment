from langgraph.graph import StateGraph, END
from core.log.logger import logger
from core.config.config import nested_config as config
from domain.enums.workflow_state import WorkflowState, RoutingDecision
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import getChatHistory

# Import all agents
from agent.small_talk_agent import SmallTalkAgent
from agent.clarification_agent import ClarificationAgent
from agent.more_detail_agent import MoreDetailAgent
from agent.refined_query_agent import RefinedQueryAgent
from agent.router_agent import RouterAgent
from agent.rag_retrieval_agent import RAGRetrievalAgent
from agent.confidence_evaluator_agent import ConfidenceEvaluatorAgent
from agent.web_search_agent import WebSearchAgent
from agent.synthesizer_agent import SynthesizerAgent
from agent.reflection_agent import ReflectionAgent

class WorkflowGraph:
    def __init__(self):
        reflection_config = config.get("reflection_agent", {})
        max_synthesis_attempts = reflection_config.get("max_synthesis_attempts", 3)

        self.agents = {
            LLMType.ROUTER_AGENT.value: RouterAgent(),
            LLMType.SMALLTALK_AGENT.value: SmallTalkAgent(),
            LLMType.CLARIFICATION_AGENT.value: ClarificationAgent(),
            LLMType.NEEDS_MORE_DETAIL_AGENT.value: MoreDetailAgent(),
            LLMType.REFINED_QUERY_AGENT.value: RefinedQueryAgent(),
            LLMType.RAG_RETRIEVAL_AGENT.value: RAGRetrievalAgent(),
            LLMType.CONFIDENCE_EVALUATOR_AGENT.value: self._create_confidence_evaluator(),
            LLMType.WEB_SEARCH_AGENT.value: WebSearchAgent(),
            LLMType.SYNTHESIZER_AGENT.value: SynthesizerAgent(),
            LLMType.REFLECTION_AGENT.value: ReflectionAgent(
                max_synthesis_attempts=max_synthesis_attempts
            ),
        }

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        workflow.add_node(LLMType.ROUTER_AGENT.value, self._router_agent)
        workflow.add_node(LLMType.CLARIFICATION_AGENT.value, self._clarification_agent)
        workflow.add_node(LLMType.SMALLTALK_AGENT.value, self._smalltalk_agent)
        workflow.add_node(LLMType.NEEDS_MORE_DETAIL_AGENT.value, self._needs_more_detail_agent)
        workflow.add_node(LLMType.REFINED_QUERY_AGENT.value, self._refined_query_agent)
        workflow.add_node(LLMType.RAG_RETRIEVAL_AGENT.value, self._rag_retrieval_agent)
        workflow.add_node(
            LLMType.CONFIDENCE_EVALUATOR_AGENT.value,
            self._confidence_evaluator_agent
        )
        workflow.add_node(LLMType.WEB_SEARCH_AGENT.value, self._web_search_agent)
        workflow.add_node(LLMType.SYNTHESIZER_AGENT.value, self._synthesizer_agent)
        workflow.add_node(LLMType.REFLECTION_AGENT.value, self._reflection_agent)

        workflow.set_entry_point(LLMType.ROUTER_AGENT.value)
        
        workflow.add_conditional_edges(
            LLMType.ROUTER_AGENT.value,
            self._route_after_router,
            {
                RoutingDecision.CLEAR_QUESTION.value: LLMType.RAG_RETRIEVAL_AGENT.value,
                RoutingDecision.AMBIGUOUS.value: LLMType.CLARIFICATION_AGENT.value,
                END: END,
            }
        )

        workflow.add_conditional_edges(
            LLMType.CLARIFICATION_AGENT.value,
            self._route_after_router,
            {
                RoutingDecision.SMALLTALK.value: LLMType.SMALLTALK_AGENT.value,
                RoutingDecision.NEEDS_MORE_DETAIL.value: LLMType.NEEDS_MORE_DETAIL_AGENT.value,
                RoutingDecision.PROCESS_QUERY.value: LLMType.REFINED_QUERY_AGENT.value,
                END: END,
            }
        )

        workflow.add_edge(LLMType.SMALLTALK_AGENT.value, END)
        workflow.add_edge(LLMType.NEEDS_MORE_DETAIL_AGENT.value, END)

        workflow.add_edge(LLMType.REFINED_QUERY_AGENT.value, LLMType.RAG_RETRIEVAL_AGENT.value)
        workflow.add_edge(LLMType.RAG_RETRIEVAL_AGENT.value, LLMType.CONFIDENCE_EVALUATOR_AGENT.value)

        workflow.add_conditional_edges(
            LLMType.CONFIDENCE_EVALUATOR_AGENT.value,
            self._route_after_confidence_evaluation,
            {
                LLMType.SYNTHESIZER_AGENT.value: LLMType.SYNTHESIZER_AGENT.value,
                LLMType.WEB_SEARCH_AGENT.value: LLMType.WEB_SEARCH_AGENT.value,
            }
        )

        workflow.add_edge(LLMType.WEB_SEARCH_AGENT.value, LLMType.SYNTHESIZER_AGENT.value)
        workflow.add_edge(LLMType.SYNTHESIZER_AGENT.value, LLMType.REFLECTION_AGENT.value)

        workflow.add_conditional_edges(
            LLMType.REFLECTION_AGENT.value,
            self._route_after_reflection,
            {
                "end": END,
                "retry": LLMType.SYNTHESIZER_AGENT.value,
            }
        )

        return workflow.compile()

    def _route_after_router(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")
        
        if routing_decision == RoutingDecision.CLEAR_QUESTION.value:
            return RoutingDecision.CLEAR_QUESTION.value
        elif routing_decision == RoutingDecision.AMBIGUOUS.value:
            return RoutingDecision.AMBIGUOUS.value
        if routing_decision == RoutingDecision.SMALLTALK.value:
            return RoutingDecision.SMALLTALK.value
        elif routing_decision == RoutingDecision.NEEDS_MORE_DETAIL.value:
            return RoutingDecision.NEEDS_MORE_DETAIL.value
        elif routing_decision == RoutingDecision.PROCESS_QUERY.value:
            return RoutingDecision.PROCESS_QUERY.value
        else:
            logger.warning(
                f"[WorkflowGraph] Unknown clarification decision: {routing_decision}"
            )
            return END

    def _route_after_confidence_evaluation(self, state: WorkflowState) -> str:
        needs_web_search = state.get("needs_web_search", False)

        if needs_web_search:
            return LLMType.WEB_SEARCH_AGENT.value
        else:
            return LLMType.SYNTHESIZER_AGENT.value
    
    def _create_confidence_evaluator(self) -> ConfidenceEvaluatorAgent:
        confidence_config = config.get("rag", {}).get("confidence_evaluation", {})

        return ConfidenceEvaluatorAgent(
            min_confidence_threshold=confidence_config.get("min_confidence_threshold", 0.6),
            min_documents_threshold=confidence_config.get("min_documents_threshold", 2),
            avg_score_threshold=confidence_config.get("avg_score_threshold", 0.5),
        )

    def _route_after_reflection(self, state: WorkflowState) -> str:
        is_sufficient = state.get("is_answer_sufficient", True)

        if is_sufficient:
            return "end"
        else:
            return "retry"

    async def _router_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[RouterAgent] Called")
        resp = await self.agents[LLMType.ROUTER_AGENT.value].invoke(state)
        return resp
    
    async def _clarification_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[ClarificationAgent] Processing user query")
        resp = await self.agents[LLMType.CLARIFICATION_AGENT.value].invoke(state)
        return resp

    async def _smalltalk_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[SmallTalkAgent] Handling casual conversation")
        resp = await self.agents[LLMType.SMALLTALK_AGENT.value].invoke(state)
        return resp

    async def _needs_more_detail_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[MoreDetailAgent] Requesting additional details")
        resp = await self.agents[LLMType.NEEDS_MORE_DETAIL_AGENT.value].invoke(state)
        return resp

    async def _refined_query_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[RefinedQueryAgent] Refining query with context")
        resp = await self.agents[LLMType.REFINED_QUERY_AGENT.value].invoke(state)
        return resp

    async def _rag_retrieval_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[RAGRetrievalAgent] Retrieving documents from knowledge base")
        resp = await self.agents[LLMType.RAG_RETRIEVAL_AGENT.value].invoke(state)
        return resp

    async def _confidence_evaluator_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[ConfidenceEvaluatorAgent] Evaluating retrieval confidence")
        resp = await self.agents[LLMType.CONFIDENCE_EVALUATOR_AGENT.value].invoke(state)
        return resp

    async def _web_search_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[WebSearchAgent] Searching web for additional information")
        resp = await self.agents[LLMType.WEB_SEARCH_AGENT.value].invoke(state)
        return resp

    async def _synthesizer_agent(self, state: WorkflowState) -> WorkflowState:
        current_attempts = state.get("synthesis_attempts", 0)
        new_attempts = current_attempts + 1

        resp = await self.agents[LLMType.SYNTHESIZER_AGENT.value].invoke(state)
        resp["synthesis_attempts"] = new_attempts

        return resp

    async def _reflection_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[ReflectionAgent] Evaluating answer quality")
        resp = await self.agents[LLMType.REFLECTION_AGENT.value].invoke(state)
        return resp

    async def invoke(self, user_input: str, session_id: str) -> WorkflowState:
        initial_state: WorkflowState = {
            "user_query": user_input,
            "session_id": session_id,
            "synthesis_attempts": 0,
        }

        chat_history = getChatHistory(session_id)
        chat_history.add_user_message(user_input)

        try:
            logger.info(f"[WorkflowGraph] Starting workflow for session: {session_id}")
            result = await self.graph.ainvoke(initial_state)

            ai_response = result.get("response", "")
            if ai_response:
                chat_history.add_ai_message(ai_response)

            quality_score = result.get('answer_quality_score', 'N/A')
            quality_str = f"{quality_score:.2f}" if isinstance(quality_score, (int, float)) else str(quality_score)
            
            logger.info(
                f"[WorkflowGraph] Workflow completed successfully. "
                f"Confidence: {result.get('confidence_score', 'N/A')}, "
                f"Quality: {quality_str}, "
                f"Synthesis attempts: {result.get('synthesis_attempts', 1)}, "
                f"Web search used: {result.get('needs_web_search', False)}"
            )

            return result

        except Exception as e:
            logger.error(f"[WorkflowGraph] Workflow error: {e}", exc_info=True)
            error_response = (
                "I apologize, but I encountered an error while processing your request. "
                "Please try again or rephrase your question."
            )

            return {
                **initial_state,
                "error_message": str(e),
                "response": error_response,
            }
