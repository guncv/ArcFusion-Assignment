from langgraph.graph import StateGraph, END
from core.log.logger import logger
from core.config.config import nested_config as config
from domain.enums.workflow_state import WorkflowState, RoutingDecision
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import getChatHistory
from infrastructure.llm.langsmith_config import LangSmithTracer
from agent.small_talk_agent import SmallTalkAgent
from agent.more_detail_agent import MoreDetailAgent
from agent.refined_query_agent import RefinedQueryAgent
from agent.clarification_agent import ClarificationAgent
from agent.intent_analysis_agent import IntentAnalysisAgent
from agent.planner_agent import PlannerAgent
from agent.tool_executor import ToolExecutor
from agent.rag_retrieval_agent import RAGRetrievalAgent
from agent.rag_synthesizer_agent import RAGSynthesizerAgent
from agent.orchestration_synthesizer_agent import OrchestrationSynthesizerAgent
from agent.rag_reflection_agent import RAGReflectionAgent
from agent.orchestration_reflection_agent import OrchestrationReflectionAgent

class WorkflowGraph:
    def __init__(self):
        self.tracer = LangSmithTracer()
        self.orchestration_max_attempts = config.get("orchestration_max_attempts", 3)

        self.agents = {
            LLMType.CLARIFICATION_AGENT.value: ClarificationAgent(),
            LLMType.SMALLTALK_AGENT.value: SmallTalkAgent(),
            LLMType.NEEDS_MORE_DETAIL_AGENT.value: MoreDetailAgent(),
            LLMType.REFINED_QUERY_AGENT.value: RefinedQueryAgent(),
            LLMType.INTENT_ANALYSIS_AGENT.value: IntentAnalysisAgent(),

            LLMType.RAG_RETRIEVAL_AGENT.value: RAGRetrievalAgent(),
            LLMType.RAG_SYNTHESIZER_AGENT.value: RAGSynthesizerAgent(),
            LLMType.RAG_REFLECTION_AGENT.value: RAGReflectionAgent(),

            LLMType.PLANNER_AGENT.value: PlannerAgent(),
            LLMType.TOOL_EXECUTOR.value: ToolExecutor(),
            LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value: OrchestrationSynthesizerAgent(),
            LLMType.ORCHESTRATION_REFLECTION_AGENT.value: OrchestrationReflectionAgent(),
        }

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        # Add all nodes
        workflow.add_node(LLMType.CLARIFICATION_AGENT.value, self._clarification_agent)
        workflow.add_node(LLMType.SMALLTALK_AGENT.value, self._smalltalk_agent)
        workflow.add_node(LLMType.NEEDS_MORE_DETAIL_AGENT.value, self._needs_more_detail_agent)
        workflow.add_node(LLMType.REFINED_QUERY_AGENT.value, self._refined_query_agent)
        workflow.add_node(LLMType.INTENT_ANALYSIS_AGENT.value, self._intent_analysis_agent)

        # RAG Agent nodes
        workflow.add_node(LLMType.RAG_AGENT.value, self._rag_agent)
        workflow.add_node(LLMType.RAG_REFLECTION_AGENT.value, self._rag_reflection_agent)

        # Orchestration nodes
        workflow.add_node(LLMType.PLANNER_AGENT.value, self._planner_agent)
        workflow.add_node(LLMType.TOOL_EXECUTOR.value, self._tool_executor)
        workflow.add_node(LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value, self._orchestration_synthesizer_agent)
        workflow.add_node(LLMType.ORCHESTRATION_REFLECTION_AGENT.value, self._orchestration_reflection_agent)

        # Set entry point
        workflow.set_entry_point(LLMType.CLARIFICATION_AGENT.value)

        # Router decides all 4 routes directly (merged init + clarification logic)
        workflow.add_conditional_edges(
            LLMType.CLARIFICATION_AGENT.value,
            self._route_after_clarification,
            {
                RoutingDecision.CLEAR_QUESTION.value: LLMType.INTENT_ANALYSIS_AGENT.value,
                RoutingDecision.SMALLTALK.value: LLMType.SMALLTALK_AGENT.value,
                RoutingDecision.NEEDS_MORE_DETAIL.value: LLMType.NEEDS_MORE_DETAIL_AGENT.value,
                RoutingDecision.PROCESS_QUERY.value: LLMType.REFINED_QUERY_AGENT.value,
                END: END,
            }
        )

        # SmallTalk and MoreDetail end the conversation
        workflow.add_edge(LLMType.SMALLTALK_AGENT.value, END)
        workflow.add_edge(LLMType.NEEDS_MORE_DETAIL_AGENT.value, END)

        # Refined query goes to intent analysis to determine RAG vs Web Search
        workflow.add_edge(LLMType.REFINED_QUERY_AGENT.value, LLMType.INTENT_ANALYSIS_AGENT.value)

        # Intent Analysis decides: use_rag -> RAG, use_web_search -> Planner (orchestration)
        workflow.add_conditional_edges(
            LLMType.INTENT_ANALYSIS_AGENT.value,
            self._route_after_intent_analysis,
            {
                RoutingDecision.USE_RAG.value: LLMType.RAG_AGENT.value,
                RoutingDecision.USE_WEB_SEARCH.value: LLMType.PLANNER_AGENT.value,
            }
        )

        # RAG Agent -> RAG Reflection
        workflow.add_edge(LLMType.RAG_AGENT.value, LLMType.RAG_REFLECTION_AGENT.value)

        # RAG Reflection decides: sufficient -> END, insufficient -> Orchestration (Planner)
        workflow.add_conditional_edges(
            LLMType.RAG_REFLECTION_AGENT.value,
            self._route_after_rag_reflection,
            {
                RoutingDecision.RAG_SUFFICIENT.value: END,
                RoutingDecision.RAG_INSUFFICIENT.value: LLMType.PLANNER_AGENT.value,
                RoutingDecision.NOT_RELEVANT.value: LLMType.PLANNER_AGENT.value,
                END: END,
            }
        )

        # Orchestration flow: Planner -> Tool Executor -> Synthesis -> Reflection
        workflow.add_edge(LLMType.PLANNER_AGENT.value, LLMType.TOOL_EXECUTOR.value)
        workflow.add_edge(LLMType.TOOL_EXECUTOR.value, LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value)
        workflow.add_edge(LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value, LLMType.ORCHESTRATION_REFLECTION_AGENT.value)

        # Orchestration Reflection decides: end or retry with replanning
        workflow.add_conditional_edges(
            LLMType.ORCHESTRATION_REFLECTION_AGENT.value,
            self._route_after_orchestration_reflection,
            {
                END: END,
                LLMType.PLANNER_AGENT.value: LLMType.PLANNER_AGENT.value,
            }
        )

        return workflow.compile()

    def _route_after_clarification(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")

        if routing_decision == RoutingDecision.CLEAR_QUESTION.value:
            logger.info("[WorkflowGraph] Router: clear_question → Intent Analysis")
            return RoutingDecision.CLEAR_QUESTION.value
        elif routing_decision == RoutingDecision.SMALLTALK.value:
            logger.info("[WorkflowGraph] Router: smalltalk → Smalltalk Agent")
            return RoutingDecision.SMALLTALK.value
        elif routing_decision == RoutingDecision.NEEDS_MORE_DETAIL.value:
            logger.info("[WorkflowGraph] Router: needs_more_detail → More Detail Agent")
            return RoutingDecision.NEEDS_MORE_DETAIL.value
        elif routing_decision == RoutingDecision.PROCESS_QUERY.value:
            logger.info("[WorkflowGraph] Router: process_query → Refined Query Agent")
            return RoutingDecision.PROCESS_QUERY.value
        else:
            logger.warning(f"[WorkflowGraph] Unknown routing decision: {routing_decision}, defaulting to needs_more_detail")
            return RoutingDecision.NEEDS_MORE_DETAIL.value

    def _route_after_intent_analysis(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")

        if routing_decision == RoutingDecision.USE_RAG.value:
            return RoutingDecision.USE_RAG.value
        elif routing_decision == RoutingDecision.USE_WEB_SEARCH.value:
            return RoutingDecision.USE_WEB_SEARCH.value
        else:
            return RoutingDecision.USE_RAG.value

    def _route_after_rag_reflection(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")

        if routing_decision == RoutingDecision.RAG_SUFFICIENT.value:
            return RoutingDecision.RAG_SUFFICIENT.value
        elif routing_decision in [RoutingDecision.RAG_INSUFFICIENT.value, RoutingDecision.NOT_RELEVANT.value]:
            return routing_decision
        else:
            return END

    def _route_after_orchestration_reflection(self, state: WorkflowState) -> str:
        is_sufficient = state.get("is_answer_sufficient", True)
        logger.info(f"[WorkflowGraph] Orchestration Reflection: response state: {state.get('response', '')}")
        logger.info(f"[WorkflowGraph] Orchestration Reflection: is_answer_sufficient: {is_sufficient}")
        logger.info(f"[WorkflowGraph] Orchestration Reflection: feedback: {state.get('reflection_issues', '')}")

        if is_sufficient :
            return END
        else:
            current_attempts = state.get("orchestration_attempts", 0) + 1
            state["orchestration_attempts"] = current_attempts
            if current_attempts >= self.orchestration_max_attempts:
                return END
            return LLMType.PLANNER_AGENT.value

    async def _clarification_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.CLARIFICATION_AGENT.value].invoke(state)
        return resp

    async def _smalltalk_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.SMALLTALK_AGENT.value].invoke(state)
        return resp

    async def _needs_more_detail_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.NEEDS_MORE_DETAIL_AGENT.value].invoke(state)
        return resp

    async def _refined_query_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.REFINED_QUERY_AGENT.value].invoke(state)
        return resp

    async def _intent_analysis_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.INTENT_ANALYSIS_AGENT.value].invoke(state)
        return resp

    async def _rag_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[WorkflowGraph] RAG Agent: state: {state}")
        rag_state = await self.agents[LLMType.RAG_RETRIEVAL_AGENT.value].invoke(state)
        synthesis_state = await self.agents[LLMType.RAG_SYNTHESIZER_AGENT.value].invoke(rag_state)
        return synthesis_state

    async def _rag_reflection_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.RAG_REFLECTION_AGENT.value].invoke(state)
        return resp

    async def _planner_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.PLANNER_AGENT.value].invoke(state)
        return resp

    async def _tool_executor(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.TOOL_EXECUTOR.value].invoke(state)
        return resp

    async def _orchestration_synthesizer_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value].invoke(state)
        return resp

    async def _orchestration_reflection_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.ORCHESTRATION_REFLECTION_AGENT.value].invoke(state)
        return resp

    async def invoke(self, user_input: str, session_id: str) -> WorkflowState:
        initial_state: WorkflowState = {
            "user_query": user_input,
            "session_id": session_id,
        }

        chat_history = getChatHistory(session_id)
        chat_history.add_user_message(user_input)

        try:
            if self.tracer.enabled and self.tracer.client:
                metadata = self.tracer.add_metadata({
                    "session_id": session_id,
                    "query_length": len(user_input),
                    "workflow_type": "rag_orchestration"
                })
                result = await self.graph.ainvoke(
                    initial_state,
                    config={"metadata": metadata}
                )
            else:
                result = await self.graph.ainvoke(initial_state)

            ai_response = result.get("response", "")
            if ai_response:
                chat_history.add_ai_message(ai_response)

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
