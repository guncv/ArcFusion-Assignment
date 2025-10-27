from src.graph.state import WorkflowState, RoutingDecision
from src.prompts import INIT_ROUTER_PROMPT
from src.constants import LLMType, LLMAgentName
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import ReActAgent
from src.infras.log import logger

class RoutingDecisionInput(BaseModel):
    decision: str = Field(description="The routing decision: 'clear_question' or 'ambiguous'")

@tool(args_schema=RoutingDecisionInput)
def finalize_routing(decision: str) -> str:
    """
    Validates and finalizes the routing decision after LLM analysis.
    This tool only validates the decision format - the LLM makes the intelligent choice.
    Use this tool to submit your final routing decision.

    Args:
        decision: The final routing decision ('clear_question' or 'ambiguous').

    Returns:
        str: Confirmation of the validated decision.
    """
    cleaned_decision = decision.strip().lower()
    valid_decisions = [RoutingDecision.CLEAR_QUESTION.value, RoutingDecision.AMBIGUOUS.value]

    if cleaned_decision not in valid_decisions:
        return "invalid_decision"
    
    return cleaned_decision

class InitRouterAgent(ReActAgent):
    def __init__(self):
        super().__init__(
            llm_type=LLMType.INIT_ROUTER_AGENT,
            system_prompt=INIT_ROUTER_PROMPT,
            tools=[finalize_routing],
            agent_name=LLMAgentName.INIT_ROUTER_AGENT.value
        )
        self.valid_decisions = [
            RoutingDecision.CLEAR_QUESTION.value,
            RoutingDecision.AMBIGUOUS.value
        ]

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            session_id = state.get("session_id", "unknown")

            logger.info(f"[{session_id}] InitRouter analyzing query: '{user_query[:100]}...'")

            # Build messages list
            messages = await self.build_message_list(
                user_query=user_query,
                include_history=False,
                state=state
            )

            # Invoke the agent
            response = await self.ainvoke_agent(messages)
            messages = response.get("messages", [])

            # Extract decision from tool result
            decision = self.extract_tool_result(
                messages=messages,
                tool_name="finalize_routing",
                valid_values=self.valid_decisions
            )

            # Fallback to default if no valid decision found
            if not decision:
                logger.warning(f"[{session_id}] InitRouter found no valid decision, defaulting to 'ambiguous'")
                decision = RoutingDecision.AMBIGUOUS.value
            else:
                logger.info(f"[{session_id}] InitRouter decision: '{decision}' for query: '{user_query[:80]}...'")

            return {
                **state,
                "routing_decision": decision,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )
