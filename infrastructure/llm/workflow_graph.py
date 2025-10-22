from langgraph.graph import StateGraph, END
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState, RoutingDecision
from agent.router_agent import RouterAgent
from domain.enums.llm_type import LLMType
from agent.small_talk import SmallTalkAgent
from agent.clarification_agent import ClarificationAgent
from agent.more_detail_agent import MoreDetailAgent
from infrastructure.llm.loader import getChatHistory
from agent.refined_query_agent import RefinedQueryAgent
class WorkflowGraph:
    
    def __init__(self):
        self.agents = {
            LLMType.ROUTER_AGENT.value: RouterAgent(),
            LLMType.SMALLTALK_AGENT.value: SmallTalkAgent(),
            LLMType.CLARIFICATION_AGENT.value: ClarificationAgent(),
            LLMType.NEEDS_MORE_DETAIL_AGENT.value: MoreDetailAgent(),
            LLMType.REFINED_QUERY_AGENT.value: RefinedQueryAgent(),
        }
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        workflow.add_node(LLMType.ROUTER_AGENT.value, self._router_agent)
        workflow.add_node(LLMType.SMALLTALK_AGENT.value, self._smalltalk_agent)
        workflow.add_node(LLMType.CLARIFICATION_AGENT.value, self._clarification_agent)
        workflow.add_node(LLMType.NEEDS_MORE_DETAIL_AGENT.value, self._needs_more_detail_agent)
        workflow.add_node(LLMType.REFINED_QUERY_AGENT.value, self._refined_query_agent)
        workflow.add_node(LLMType.INTENT_ANALYSIS_AGENT.value, self._intent_analysis_agent)
        
        workflow.set_entry_point(LLMType.ROUTER_AGENT.value)
        
        workflow.add_conditional_edges(
            LLMType.ROUTER_AGENT.value,
            self._route_after_router,
            {
                LLMType.INTENT_ANALYSIS_AGENT.value: LLMType.INTENT_ANALYSIS_AGENT.value,
                LLMType.CLARIFICATION_AGENT.value: LLMType.CLARIFICATION_AGENT.value,
                END: END,
            }
        )

        workflow.add_conditional_edges(
            LLMType.CLARIFICATION_AGENT.value,
            self._route_after_router,
            {
                LLMType.SMALLTALK_AGENT.value: LLMType.SMALLTALK_AGENT.value,
                LLMType.NEEDS_MORE_DETAIL_AGENT.value: LLMType.NEEDS_MORE_DETAIL_AGENT.value,
                LLMType.REFINED_QUERY_AGENT.value: LLMType.REFINED_QUERY_AGENT.value,
                END: END,
            }
        )

        workflow.add_edge(LLMType.SMALLTALK_AGENT.value, END)
        workflow.add_edge(LLMType.NEEDS_MORE_DETAIL_AGENT.value, END)
        workflow.add_edge(LLMType.REFINED_QUERY_AGENT.value, LLMType.INTENT_ANALYSIS_AGENT.value)
        workflow.add_edge(LLMType.INTENT_ANALYSIS_AGENT.value, END)
        
        return workflow.compile()
    
    def _route_after_router(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")

        if routing_decision == RoutingDecision.CLEAR_QUESTION.value:
            return LLMType.INTENT_ANALYSIS_AGENT.value
        elif routing_decision == RoutingDecision.AMBIGUOUS.value:
            return LLMType.CLARIFICATION_AGENT.value
        elif routing_decision == RoutingDecision.SMALLTALK.value:
            return LLMType.SMALLTALK_AGENT.value
        elif routing_decision == RoutingDecision.NEEDS_MORE_DETAIL.value:
            return LLMType.NEEDS_MORE_DETAIL_AGENT.value
        elif routing_decision == RoutingDecision.PROCESS_QUERY.value:
            return LLMType.REFINED_QUERY_AGENT.value
        else:
            return END

    async def _router_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[RouterAgent] Called")
        resp = await self.agents[LLMType.ROUTER_AGENT.value].invoke(state)
        return resp
    
    async def _smalltalk_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[SmallTalkAgent] Called")
        resp = await self.agents[LLMType.SMALLTALK_AGENT.value].invoke(state)
        return resp
    
    async def _intent_analysis_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[IntentAnalysisAgent] Called")
        return state
    
    async def _clarification_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[ClarificationAgent] Called")
        resp = await self.agents[LLMType.CLARIFICATION_AGENT.value].invoke(state)
        return resp
    
    async def _needs_more_detail_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[NeedsMoreDetailAgent] Called")
        resp = await self.agents[LLMType.NEEDS_MORE_DETAIL_AGENT.value].invoke(state)
        return resp

    async def _refined_query_agent(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[RefinedQueryAgent] Called")
        resp = await self.agents[LLMType.REFINED_QUERY_AGENT.value].invoke(state)
        return resp

    async def invoke(self, user_input: str, session_id: str) -> WorkflowState:
        initial_state: WorkflowState = {
            "user_query": user_input,
            "session_id": session_id
        }

        chat_history = getChatHistory(session_id)
        chat_history.add_user_message(user_input)

        try:
            result = await self.graph.ainvoke(initial_state)

            ai_response = result.get("response", "")
            if ai_response:
                chat_history.add_ai_message(ai_response)
            return result
        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            return {
                **initial_state,
                "error_message": str(e),
                "response": "An error occurred while processing your request."
            }