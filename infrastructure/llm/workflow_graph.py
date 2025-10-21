from langgraph.graph import StateGraph, END
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from agent.router_agent import RouterAgent
from domain.enums.llm_type import LLMType

class WorkflowGraph:
    
    def __init__(self):
        logger.info("Initializing WorkflowGraph")

        self.router_agent = RouterAgent()

        self.graph = self._build_graph()
        logger.info("WorkflowGraph initialized successfully")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        workflow.add_node(LLMType.ROUTER_AGENT.value, self._router_agent)
        workflow.set_entry_point(LLMType.ROUTER_AGENT.value)
        workflow.add_edge(LLMType.ROUTER_AGENT.value, END)
        return workflow.compile()

    async def _router_agent(self, state: WorkflowState) -> WorkflowState:
        resp = await self.router_agent.invoke(state)
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
                "final_response": "An error occurred while processing your request."
            }