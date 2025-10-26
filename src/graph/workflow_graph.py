from langgraph.graph import StateGraph, END
from src.config import config
from src.graph.state import WorkflowState, RoutingDecision
from src.constants import LLMType
from src.infras.log import langsmith_tracer
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
        self.tracer = langsmith_tracer
        self.orchestration_max_attempts = config.get("orchestration_max_attempts", 2)

        self.agents = {
            # Initial routing phase
            LLMType.INIT_ROUTER_AGENT.value: InitRouterAgent(),

            # Clarification phase
            LLMType.CLARIFICATION_AGENT.value: ClarificationAgent(),
            LLMType.SMALLTALK_AGENT.value: SmallTalkAgent(),
            LLMType.NEEDS_MORE_DETAIL_AGENT.value: MoreDetailAgent(),
            LLMType.REFINED_QUERY_AGENT.value: RefinedQueryAgent(),

            # Orchestration phase (autonomous tool selection)
            LLMType.PLANNER_AGENT.value: PlannerAgent(),
            LLMType.TOOL_EXECUTOR.value: ToolExecutor(),
            LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value: SynthesizerAgent(),
            LLMType.ORCHESTRATION_REFLECTION_AGENT.value: ReflectionAgent(),
        }

        self.graph = self._build_graph()

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
        workflow.add_node(LLMType.ORCHESTRATION_SYNTHESIZER_AGENT.value, self._orchestration_synthesizer_agent)
        workflow.add_node(LLMType.ORCHESTRATION_REFLECTION_AGENT.value, self._orchestration_reflection_agent)

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

    def _route_after_init_router(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")

        if routing_decision == RoutingDecision.CLEAR_QUESTION.value:
            return RoutingDecision.CLEAR_QUESTION.value
        elif routing_decision == RoutingDecision.AMBIGUOUS.value:
            return RoutingDecision.AMBIGUOUS.value
        else:
            return RoutingDecision.NEEDS_MORE_DETAIL.value

    def _route_after_clarification(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")

        if routing_decision == RoutingDecision.SMALLTALK.value:
            return RoutingDecision.SMALLTALK.value
        elif routing_decision == RoutingDecision.NEEDS_MORE_DETAIL.value:
            return RoutingDecision.NEEDS_MORE_DETAIL.value
        elif routing_decision == RoutingDecision.PROCESS_QUERY.value:
            return RoutingDecision.PROCESS_QUERY.value
        else:
            return RoutingDecision.NEEDS_MORE_DETAIL.value

    def _route_after_orchestration_reflection(self, state: WorkflowState) -> str:
        is_sufficient = state.get("is_answer_sufficient", True)
        if is_sufficient:
            return END
        else:
            # orchestration_attempts is now incremented in orchestration_reflection_agent
            # (state mutations in routing functions don't persist in LangGraph)
            current_attempts = state.get("orchestration_attempts", 0)
            if current_attempts >= self.orchestration_max_attempts:
                return END
            return LLMType.PLANNER_AGENT.value

    async def _init_router_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.agents[LLMType.INIT_ROUTER_AGENT.value].invoke(state)
        return resp

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

    async def ainvoke(self, user_input: str, session_id: str) -> WorkflowState:
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
            error_response = (
                "I apologize, but I encountered an error while processing your request. "
                "Please try again or rephrase your question."
            )

            return {
                **initial_state,
                "error_message": str(e),
                "response": error_response,
            }

workflow_graph = WorkflowGraph()