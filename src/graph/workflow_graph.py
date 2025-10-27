from langgraph.graph import StateGraph, END
from src.config import config
from src.graph.state import WorkflowState, RoutingDecision
from src.constants import LLMType
from src.infras.log import langsmith_tracer, logger
from src.repositories.chat_history import get_chat_history_repository
import traceback
from src.agent import (
    SmallTalkAgent,
    MoreDetailAgent,
    RefinedQueryAgent,
    ClarificationAgent,
    InitRouterAgent,
    PlannerAgent,
    ToolExecutor,
    SynthesizerAgent,
    ReflectionAgent
)

class WorkflowGraph:
    def __init__(self):
        logger.info("Initializing WorkflowGraph")
        self.tracer = langsmith_tracer
        self.orchestration_max_attempts = config.get("orchestration_max_attempts", 2)
        self.chat_history_repo = get_chat_history_repository()

        logger.debug("Instantiating workflow agents")
        self.agents = {
            LLMType.INIT_ROUTER_AGENT.value: InitRouterAgent(),
            LLMType.CLARIFICATION_AGENT.value: ClarificationAgent(),
            LLMType.SMALLTALK_AGENT.value: SmallTalkAgent(),
            LLMType.NEEDS_MORE_DETAIL_AGENT.value: MoreDetailAgent(),
            LLMType.REFINED_QUERY_AGENT.value: RefinedQueryAgent(),
            LLMType.PLANNER_AGENT.value: PlannerAgent(),
            LLMType.TOOL_EXECUTOR.value: ToolExecutor(),
            LLMType.SYNTHESIZER_AGENT.value: SynthesizerAgent(),
            LLMType.REFLECTION_AGENT.value: ReflectionAgent(),
        }
        logger.info(f"Initialized {len(self.agents)} agents successfully")

        self.graph = self._build_graph()
        logger.info("WorkflowGraph built successfully")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        # Add all nodes
        workflow.add_node(LLMType.INIT_ROUTER_AGENT.value, self._init_router_agent)
        workflow.add_node(LLMType.CLARIFICATION_AGENT.value, self._clarification_agent)
        workflow.add_node(LLMType.SMALLTALK_AGENT.value, self._smalltalk_agent)
        workflow.add_node(LLMType.NEEDS_MORE_DETAIL_AGENT.value, self._needs_more_detail_agent)
        workflow.add_node(LLMType.REFINED_QUERY_AGENT.value, self._refined_query_agent)

        # Orchestration nodes (autonomous tool selection via Planner)
        workflow.add_node(LLMType.PLANNER_AGENT.value, self._planner_agent)
        workflow.add_node(LLMType.TOOL_EXECUTOR.value, self._tool_executor)
        workflow.add_node(LLMType.SYNTHESIZER_AGENT.value, self._synthesizer_agent)
        workflow.add_node(LLMType.REFLECTION_AGENT.value, self._reflection_agent)

        # Set entry point
        workflow.set_entry_point(LLMType.INIT_ROUTER_AGENT.value)
        
        workflow.add_conditional_edges(
            LLMType.INIT_ROUTER_AGENT.value,
            self._route_after_init_router,
            {
                RoutingDecision.CLEAR_QUESTION.value: LLMType.PLANNER_AGENT.value,
                RoutingDecision.AMBIGUOUS.value: LLMType.CLARIFICATION_AGENT.value,
            }
        )

        # Clarification agent routes to 3 possible destinations
        workflow.add_conditional_edges(
            LLMType.CLARIFICATION_AGENT.value,
            self._route_after_clarification,
            {
                RoutingDecision.SMALLTALK.value: LLMType.SMALLTALK_AGENT.value,
                RoutingDecision.NEEDS_MORE_DETAIL.value: LLMType.NEEDS_MORE_DETAIL_AGENT.value,
                RoutingDecision.PROCESS_QUERY.value: LLMType.REFINED_QUERY_AGENT.value,
            }
        )

        # SmallTalk and MoreDetail end the conversation
        workflow.add_edge(LLMType.SMALLTALK_AGENT.value, END)
        workflow.add_edge(LLMType.NEEDS_MORE_DETAIL_AGENT.value, END)

        # Refined query goes directly to Planner (autonomous tool selection)
        workflow.add_edge(LLMType.REFINED_QUERY_AGENT.value, LLMType.PLANNER_AGENT.value)

        # Orchestration flow: Planner -> Tool Executor -> Synthesis -> Reflection
        workflow.add_edge(LLMType.PLANNER_AGENT.value, LLMType.TOOL_EXECUTOR.value)
        workflow.add_edge(LLMType.TOOL_EXECUTOR.value, LLMType.SYNTHESIZER_AGENT.value)
        workflow.add_edge(LLMType.SYNTHESIZER_AGENT.value, LLMType.REFLECTION_AGENT.value)

        # Orchestration Reflection decides: end or retry with replanning
        workflow.add_conditional_edges(
            LLMType.REFLECTION_AGENT.value,
            self._route_after_reflection,
            {
                END: END,
                LLMType.PLANNER_AGENT.value: LLMType.PLANNER_AGENT.value,
            }
        )

        return workflow.compile()

    def _route_after_init_router(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")
        logger.info(f"InitRouter decision: {routing_decision}")

        if routing_decision == RoutingDecision.CLEAR_QUESTION.value:
            logger.debug("Routing to Planner (clear question)")
            return RoutingDecision.CLEAR_QUESTION.value
        elif routing_decision == RoutingDecision.AMBIGUOUS.value:
            logger.debug("Routing to Clarification (ambiguous question)")
            return RoutingDecision.AMBIGUOUS.value
        else:
            logger.debug("Routing to NeedsMoreDetail (fallback)")
            return RoutingDecision.NEEDS_MORE_DETAIL.value

    def _route_after_clarification(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")
        logger.info(f"Clarification decision: {routing_decision}")

        if routing_decision == RoutingDecision.SMALLTALK.value:
            logger.debug("Routing to SmallTalk")
            return RoutingDecision.SMALLTALK.value
        elif routing_decision == RoutingDecision.NEEDS_MORE_DETAIL.value:
            logger.debug("Routing to NeedsMoreDetail")
            return RoutingDecision.NEEDS_MORE_DETAIL.value
        elif routing_decision == RoutingDecision.PROCESS_QUERY.value:
            logger.debug("Routing to RefinedQuery")
            return RoutingDecision.PROCESS_QUERY.value
        else:
            logger.debug("Routing to NeedsMoreDetail (fallback)")
            return RoutingDecision.NEEDS_MORE_DETAIL.value

    def _route_after_reflection(self, state: WorkflowState) -> str:
        is_sufficient = state.get("is_answer_sufficient", True)
        current_attempts = state.get("orchestration_attempts", 0)
        logger.info(f"Reflection decision: sufficient={is_sufficient}, attempts={current_attempts}/{self.orchestration_max_attempts}")

        if is_sufficient:
            logger.debug("Answer sufficient, ending workflow")
            return END
        else:
            if current_attempts >= self.orchestration_max_attempts:
                logger.warning(f"Max orchestration attempts ({self.orchestration_max_attempts}) reached, ending workflow")
                return END
            logger.debug(f"Retrying with Planner (attempt {current_attempts + 1})")
            return LLMType.PLANNER_AGENT.value

    async def _init_router_agent(self, state: WorkflowState) -> WorkflowState:
        logger.debug("Invoking InitRouterAgent")
        resp = await self.agents[LLMType.INIT_ROUTER_AGENT.value].ainvoke(state)
        logger.debug(f"InitRouterAgent completed: routing_decision={resp.get('routing_decision')}")
        return resp

    async def _clarification_agent(self, state: WorkflowState) -> WorkflowState:
        logger.debug("Invoking ClarificationAgent")
        resp = await self.agents[LLMType.CLARIFICATION_AGENT.value].ainvoke(state)
        logger.debug(f"ClarificationAgent completed")
        return resp

    async def _smalltalk_agent(self, state: WorkflowState) -> WorkflowState:
        logger.debug("Invoking SmallTalkAgent")
        resp = await self.agents[LLMType.SMALLTALK_AGENT.value].ainvoke(state)
        logger.debug("SmallTalkAgent completed")
        return resp

    async def _needs_more_detail_agent(self, state: WorkflowState) -> WorkflowState:
        logger.debug("Invoking NeedsMoreDetailAgent")
        resp = await self.agents[LLMType.NEEDS_MORE_DETAIL_AGENT.value].ainvoke(state)
        logger.debug("NeedsMoreDetailAgent completed")
        return resp

    async def _refined_query_agent(self, state: WorkflowState) -> WorkflowState:
        logger.debug("Invoking RefinedQueryAgent")
        resp = await self.agents[LLMType.REFINED_QUERY_AGENT.value].ainvoke(state)
        logger.debug(f"RefinedQueryAgent completed: refined_query={resp.get('refined_query', '')[:100]}")
        return resp

    async def _planner_agent(self, state: WorkflowState) -> WorkflowState:
        attempt = state.get("orchestration_attempts", 0)
        logger.info(f"Invoking PlannerAgent (attempt {attempt + 1})")
        resp = await self.agents[LLMType.PLANNER_AGENT.value].ainvoke(state)
        logger.info(f"PlannerAgent completed: tool={resp.get('selected_tool')}, queries={len(resp.get('generated_queries', []))}")
        return resp

    async def _tool_executor(self, state: WorkflowState) -> WorkflowState:
        tool = state.get("selected_tool")
        logger.info(f"Invoking ToolExecutor with tool={tool}")
        resp = await self.agents[LLMType.TOOL_EXECUTOR.value].ainvoke(state)
        logger.info(f"ToolExecutor completed")
        return resp

    async def _synthesizer_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("Invoking SynthesizerAgent")
        resp = await self.agents[LLMType.SYNTHESIZER_AGENT.value].ainvoke(state)
        response_len = len(resp.get("response", ""))
        logger.info(f"SynthesizerAgent completed: response_len={response_len}")
        return resp

    async def _reflection_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info("Invoking ReflectionAgent")
        resp = await self.agents[LLMType.REFLECTION_AGENT.value].ainvoke(state)
        logger.info(f"ReflectionAgent completed: sufficient={resp.get('is_answer_sufficient')}")
        return resp

    async def ainvoke(self, user_input: str, session_id: str) -> WorkflowState:
        logger.info(f"[{session_id}] Starting workflow for query (len={len(user_input)}): {user_input[:100]}...")
        initial_state: WorkflowState = {
            "user_query": user_input,
            "session_id": session_id,
        }

        await self.chat_history_repo.add_message(
            session_id=session_id,
            message_type="human",
            content=user_input
        )
        logger.debug(f"[{session_id}] Saved user message to chat history")

        try:
            if self.tracer.enabled and self.tracer.client:
                logger.debug(f"[{session_id}] LangSmith tracing enabled")
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
                logger.debug(f"[{session_id}] LangSmith tracing disabled")
                result = await self.graph.ainvoke(initial_state)

            ai_response = result.get("response", "")
            if ai_response:
                logger.info(f"[{session_id}] Workflow completed successfully (response_len={len(ai_response)})")
                await self.chat_history_repo.add_message(
                    session_id=session_id,
                    message_type="ai",
                    content=ai_response
                )

            return result

        except Exception as e:
            error_response = (
                "I apologize, but I encountered an error while processing your request. "
                "Please try again or rephrase your question."
            )

            # Log the full error details for debugging
            logger.error(f"Workflow error for session {session_id}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

            return {
                **initial_state,
                "error_message": str(e),
                "response": error_response,
            }

workflow_graph = WorkflowGraph()