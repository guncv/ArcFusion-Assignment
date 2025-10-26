from src.graph.state import WorkflowState, RoutingDecision
from prompts import CLARIFICATION_PROMPT
from src.constants import LLMType, LLMAgentName
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import react_agent

class ClarificationDecisionInput(BaseModel):
    decision: str = Field(description="The routing decision: 'smalltalk', 'needs_more_detail', or 'process_query'")

@tool(args_schema=ClarificationDecisionInput)
def finalize_clarification_routing(decision: str) -> str:
    """
    Validates and finalizes the clarification routing decision after LLM analysis.
    This tool only validates the decision format - the LLM makes the intelligent choice.
    Use this tool to submit your final clarification routing decision.

    Args:
        decision: The final routing decision ('smalltalk', 'needs_more_detail', or 'process_query').

    Returns:
        str: Confirmation of the validated decision.
    """
    cleaned_decision = decision.strip().lower()
    valid_decisions = [
        RoutingDecision.SMALLTALK.value,
        RoutingDecision.NEEDS_MORE_DETAIL.value,
        RoutingDecision.PROCESS_QUERY.value,
    ]

    if cleaned_decision not in valid_decisions:
        return "invalid_decision"

    return cleaned_decision

class ClarificationAgent(react_agent):
    def __init__(self):
        super().__init__(
            llm_type=LLMType.CLARIFICATION_AGENT,
            system_prompt=CLARIFICATION_PROMPT,
            tools=[finalize_clarification_routing],
            agent_name=LLMAgentName.CLARIFICATION_AGENT.value
        )
        self.valid_decisions = [
            RoutingDecision.SMALLTALK.value,
            RoutingDecision.NEEDS_MORE_DETAIL.value,
            RoutingDecision.PROCESS_QUERY.value,
        ]

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            
            # Build messages list with history + current query
            messages = self.build_message_list(
                user_query=user_query,
                include_history=True,
                state=state
            )
            
            # Invoke the agent
            response = await self.ainvoke_agent(messages)
            
            # Extract the routing decision from tool calls
            messages = response.get("messages", [])
            
            # Extract decision from tool result
            decision = self.extract_tool_result(
                messages=messages,
                tool_name="finalize_clarification_routing",
                valid_values=self.valid_decisions
            )
            
            # Fallback to default if no valid decision found
            if not decision:
                decision = RoutingDecision.NEEDS_MORE_DETAIL.value
            
            return {
                **state,
                "routing_decision": decision,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

clarification_agent = ClarificationAgent()