from langgraph.graph import StateGraph, END
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState, RoutingDecision
from agent.router_agent import RouterAgent
from domain.enums.llm_type import LLMType
from agent.small_talk import SmallTalkAgent

class WorkflowGraph:
    
    def __init__(self):
        self.router_agent = RouterAgent()
        self.smalltalk_agent = SmallTalkAgent()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        workflow.add_node(LLMType.ROUTER_AGENT.value, self._router_agent)
        workflow.add_node(LLMType.SMALLTALK_AGENT.value, self._smalltalk_agent)
        
        workflow.set_entry_point(LLMType.ROUTER_AGENT.value)
        
        workflow.add_conditional_edges(
            LLMType.ROUTER_AGENT.value,
            self._route_after_router,
            {
                LLMType.SMALLTALK_AGENT.value: LLMType.SMALLTALK_AGENT.value,
                END: END
            }
        )
        
        workflow.add_edge(LLMType.SMALLTALK_AGENT.value, END)
        
        return workflow.compile()
    
    def _route_after_router(self, state: WorkflowState) -> str:
        routing_decision = state.get("routing_decision", "")
        logger.info(f"Routing decision: {routing_decision}")
        
        if routing_decision == RoutingDecision.SMALLTALK.value:
            return LLMType.SMALLTALK_AGENT.value
        else:
            return END

    async def _router_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.router_agent.invoke(state)
        return resp
    
    async def _smalltalk_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.smalltalk_agent.invoke(state)
        return resp

    async def invoke(self, user_input: str) -> WorkflowState:
        logger.info(f"Invoking workflow with input: {user_input}")

        initial_state: WorkflowState = {
            "user_query": user_input
        }

        try:
            result = await self.graph.ainvoke(initial_state)
            logger.info("Workflow completed successfully")
            return result
        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            return {
                **initial_state,
                "error_message": str(e),
                "response": "An error occurred while processing your request."
            }