from langgraph.graph import StateGraph, END
from core.log.logger import logger
from core.config.config import nested_config as config
from domain.enums.workflow_state import WorkflowState, RoutingDecision
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import getChatHistory
from infrastructure.llm.langsmith_config import LangSmithTracer
from agent.small_talk_agent import SmallTalkAgent
from agent.clarification_agent import ClarificationAgent
from agent.more_detail_agent import MoreDetailAgent
from agent.refined_query_agent import RefinedQueryAgent
from agent.intent_router_agent import IntentRouterAgent
from agent.planner_agent import PlannerAgent
from agent.tool_executor import ToolExecutor
from agent.rag_retrieval_agent import RAGRetrievalAgent
from agent.web_search_agent import WebSearchAgent
from agent.rag_synthesizer_agent import RAGSynthesizerAgent
from agent.orchestration_synthesizer_agent import OrchestrationSynthesizerAgent
from agent.rag_reflection_agent import RAGReflectionAgent
from agent.orchestration_reflection_agent import OrchestrationReflectionAgent

class WorkflowGraph:
    def __init__(self):
        orchestration_reflection_config = config.get("orchestration_reflection_agent", {})
        max_synthesis_attempts = orchestration_reflection_config.get("max_synthesis_attempts", 3)
        self.tracer = LangSmithTracer()
        self.agents = {
            LLMType.INTENT_ROUTER_AGENT.value: IntentRouterAgent(),
            LLMType.SMALLTALK_AGENT.value: SmallTalkAgent(),
            LLMType.CLARIFICATION_AGENT.value: ClarificationAgent(),
            LLMType.NEEDS_MORE_DETAIL_AGENT.value: MoreDetailAgent(),
            LLMType.REFINED_QUERY_AGENT.value: RefinedQueryAgent(),
            
            LLMType.RAG_RETRIEVAL_AGENT.value: RAGRetrievalAgent(),
            LLMType.RAG_SYNTHESIZER_AGENT.value: RAGSynthesizerAgent(),
            LLMType.RAG_REFLECTION_AGENT.value: RAGReflectionAgent(),
            
            LLMType.PLANNER_AGENT.value: PlannerAgent(),
            LLMType.WEB_SEARCH_AGENT.value: WebSearchAgent(),
            LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value: OrchestrationSynthesizerAgent(),
            LLMType.ORCHESTRATION_REFLECTION_AGENT.value: OrchestrationReflectionAgent(
                max_synthesis_attempts=max_synthesis_attempts
            ),
        }

        self.tool_executor = ToolExecutor(self.agents)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        # Add all nodes
        workflow.add_node(LLMType.INTENT_ROUTER_AGENT.value, self._intent_router_agent)
        workflow.add_node(LLMType.CLARIFICATION_AGENT.value, self._clarification_agent)
        workflow.add_node(LLMType.SMALLTALK_AGENT.value, self._smalltalk_agent)
        workflow.add_node(LLMType.NEEDS_MORE_DETAIL_AGENT.value, self._needs_more_detail_agent)
        workflow.add_node(LLMType.REFINED_QUERY_AGENT.value, self._refined_query_agent)

        # RAG Agent nodes (Fixed Part - always executes)
        workflow.add_node(LLMType.RAG_AGENT.value, self._rag_agent)  # Combined RAG retrieval + synthesis
        workflow.add_node(LLMType.RAG_REFLECTION_AGENT.value, self._rag_reflection_agent)

        # Orchestration nodes (Dynamic Part - only if RAG insufficient)
        workflow.add_node(LLMType.PLANNER_AGENT.value, self._planner_agent)
        workflow.add_node(LLMType.TOOL_EXECUTOR.value, self._tool_executor)
        workflow.add_node(LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value, self._orchestration_synthesizer_agent)
        workflow.add_node(LLMType.ORCHESTRATION_REFLECTION_AGENT.value, self._orchestration_reflection_agent)

        # Set entry point
        workflow.set_entry_point(LLMType.INTENT_ROUTER_AGENT.value)

        # Router decides: clear question -> RAG, ambiguous -> clarification
        workflow.add_conditional_edges(
            LLMType.INTENT_ROUTER_AGENT.value,
            self._route_after_intent_router,
            {
                RoutingDecision.CLEAR_QUESTION.value: LLMType.RAG_AGENT.value,
                RoutingDecision.AMBIGUOUS.value: LLMType.CLARIFICATION_AGENT.value,
                END: END,
            }
        )

        # Clarification decides: smalltalk, needs more detail, or refine query
        workflow.add_conditional_edges(
            LLMType.CLARIFICATION_AGENT.value,
            self._route_after_clarification,
            {
                RoutingDecision.SMALLTALK.value: LLMType.SMALLTALK_AGENT.value,
                RoutingDecision.NEEDS_MORE_DETAIL.value: LLMType.NEEDS_MORE_DETAIL_AGENT.value,
                RoutingDecision.PROCESS_QUERY.value: LLMType.REFINED_QUERY_AGENT.value,
                END: END,
            }
        )

        # SmallTalk and MoreDetail end the conversation
        workflow.add_edge(LLMType.SMALLTALK_AGENT.value, END)
        workflow.add_edge(LLMType.NEEDS_MORE_DETAIL_AGENT.value, END)

        # Refined query goes to RAG agent (not planner!)
        workflow.add_edge(LLMType.REFINED_QUERY_AGENT.value, LLMType.RAG_AGENT.value)

        # RAG Agent -> RAG Reflection
        workflow.add_edge(LLMType.RAG_AGENT.value, LLMType.RAG_REFLECTION_AGENT.value)

        # RAG Reflection decides: sufficient -> END, insufficient -> Orchestration (Planner)
        workflow.add_conditional_edges(
            LLMType.RAG_REFLECTION_AGENT.value,
            self._route_after_rag_reflection,
            {
                "end": END,
                "orchestration": LLMType.PLANNER_AGENT.value,
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
                "end": END,
                "retry": LLMType.PLANNER_AGENT.value,
            }
        )

        return workflow.compile()

    def _route_after_intent_router(self, state: WorkflowState) -> str:
        """Route after the router agent."""
        routing_decision = state.get("routing_decision", "")

        if routing_decision == RoutingDecision.CLEAR_QUESTION.value:
            return RoutingDecision.CLEAR_QUESTION.value
        elif routing_decision == RoutingDecision.AMBIGUOUS.value:
            return RoutingDecision.AMBIGUOUS.value
        else:
            logger.warning(
                f"[WorkflowGraph] Unknown routing decision: {routing_decision}, defaulting to ambiguous"
            )
            return RoutingDecision.AMBIGUOUS.value

    def _route_after_clarification(self, state: WorkflowState) -> str:
        """Route after the clarification agent."""
        routing_decision = state.get("routing_decision", "")

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

    def _route_after_rag_reflection(self, state: WorkflowState) -> str:
        """Route after RAG reflection - decide if RAG alone is sufficient."""
        is_rag_sufficient = state.get("is_rag_sufficient", False)

        if is_rag_sufficient:
            logger.info("[WorkflowGraph] RAG results sufficient - ending workflow")
            return "end"
        else:
            logger.info("[WorkflowGraph] RAG results insufficient - moving to orchestration")
            return "orchestration"

    def _route_after_orchestration_reflection(self, state: WorkflowState) -> str:
        is_sufficient = state.get("is_answer_sufficient", True)

        if is_sufficient:
            logger.info("[WorkflowGraph] Orchestration complete - ending workflow")
            return "end"
        else:
            logger.info("[WorkflowGraph] Orchestration insufficient - replanning")
            return "retry"

    async def _intent_router_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[RouterAgent] Called")
        resp = await self.agents[LLMType.INTENT_ROUTER_AGENT.value].invoke(state)
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

    async def _rag_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[RAG Agent] Starting RAG retrieval and synthesis")

        rag_state = await self.agents[LLMType.RAG_RETRIEVAL_AGENT.value].invoke(state)
        synthesis_state = await self.agents[LLMType.RAG_SYNTHESIZER_AGENT.value].invoke(rag_state)

        logger.info(
            f"[RAG Agent] Complete - "
            f"Retrieved: {len(synthesis_state.get('retrieved_documents', []))} docs, "
            f"Confidence: {synthesis_state.get('confidence_score', 0.0):.2f}"
        )

        return synthesis_state

    async def _rag_reflection_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("[RAG Reflection Agent] Evaluating RAG sufficiency")
        resp = await self.agents[LLMType.RAG_REFLECTION_AGENT.value].invoke(state)
        return resp

    async def _planner_agent(self, state: WorkflowState) -> WorkflowState:
        # Track planning attempts
        planning_attempts = state.get("planning_attempts", 0) + 1
        previous_attempts = state.get("previous_attempts", [])

        # Check if this is a retry
        is_retry = planning_attempts > 1

        if is_retry:
            logger.info(f"[PlannerAgent] Replanning (attempt {planning_attempts})")
            logger.info(f"[PlannerAgent] Previous attempts: {len(previous_attempts)}")

            # Log what was tried before
            if previous_attempts:
                last_attempt = previous_attempts[-1]
                logger.info(f"[PlannerAgent] Previous tools: {last_attempt.get('tools', [])}")
                logger.info(f"[PlannerAgent] Previous confidence: {last_attempt.get('confidence', 'N/A')}")
        else:
            logger.info("[PlannerAgent] Creating initial execution plan")

        # Update state with attempt tracking before calling planner
        state_with_attempts = {
            **state,
            "planning_attempts": planning_attempts,
            "is_replanning": is_retry
        }

        resp = await self.agents[LLMType.PLANNER_AGENT.value].invoke(state_with_attempts)

        # Save this attempt to history
        current_attempt = {
            "attempt": planning_attempts,
            "tools": resp.get("planned_tools", []),
            "plan": resp.get("execution_plan", ""),
            "confidence": state.get("confidence_score", 0.0),
            "tool_results": state.get("tool_results", {})
        }

        previous_attempts.append(current_attempt)

        resp["planning_attempts"] = planning_attempts
        resp["previous_attempts"] = previous_attempts

        logger.info(f"[PlannerAgent] Plan: {resp.get('execution_plan', 'N/A')}")
        logger.info(f"[PlannerAgent] Tools: {resp.get('planned_tools', [])}")

        return resp

    async def _tool_executor(self, state: WorkflowState) -> WorkflowState:
        logger.info("[ToolExecutor] Executing planned tools")
        resp = await self.tool_executor.invoke(state)
        return resp

    async def _orchestration_synthesizer_agent(self, state: WorkflowState) -> WorkflowState:
        """Synthesize the final answer by merging RAG and Web search results."""
        logger.info("[Orchestration Synthesizer] Merging RAG + Web results")

        current_attempts = state.get("synthesis_attempts", 0)
        new_attempts = current_attempts + 1

        # Use specialized orchestration synthesizer to merge RAG + Web
        resp = await self.agents[LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value].invoke(state)
        resp["synthesis_attempts"] = new_attempts

        return resp

    async def _orchestration_reflection_agent(self, state: WorkflowState) -> WorkflowState:
        """Evaluate final answer quality after orchestration."""
        logger.info("[Orchestration Reflection Agent] Evaluating final answer quality")
        resp = await self.agents[LLMType.ORCHESTRATION_REFLECTION_AGENT.value].invoke(state)
        return resp

    async def invoke(self, user_input: str, session_id: str) -> WorkflowState:
        initial_state: WorkflowState = {
            "user_query": user_input,
            "session_id": session_id,
            "synthesis_attempts": 0,
            "planning_attempts": 0,
            "previous_attempts": [],
        }

        chat_history = getChatHistory(session_id)
        chat_history.add_user_message(user_input)

        try:
            logger.info(f"[WorkflowGraph] Starting workflow for session: {session_id}")

            # Add LangSmith metadata if tracing is enabled
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
