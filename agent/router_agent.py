from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState, RoutingDecision
from prompts.router_agent_prompt import ROUTER_AGENT_PROMPT
from infrastructure.llm.loader import loadLLM
from domain.enums.llm_type import LLMType
from langchain_core.tools import tool
from langchain.agents import create_agent
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from pydantic import BaseModel, Field

class RoutingDecisionInput(BaseModel):
    decision: str = Field(description="The routing decision: 'clear_question' or 'ambiguous'")

@tool(args_schema=RoutingDecisionInput)
def finalize_routing(decision: str) -> str:
    """
    Finalizes the routing decision after validation.
    Use this tool to submit your final routing decision.

    Args:
        decision: The final routing decision ('clear_question' or 'ambiguous').

    Returns:
        str: Confirmation of the finalized decision.
    """
    cleaned_decision = decision.strip().lower()
    valid_decisions = [RoutingDecision.CLEAR_QUESTION.value, RoutingDecision.AMBIGUOUS.value]

    if cleaned_decision not in valid_decisions:
        logger.error(f"[Tool - finalize_routing] Attempted to finalize invalid decision: {decision}")
        return "invalid_decision"

    return cleaned_decision

class RouterAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.ROUTER_AGENT)
        self.tools = [finalize_routing]
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=ROUTER_AGENT_PROMPT
        )

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")

            # Step 1: Run the agent
            agent_input = {"messages": [{"role": "user", "content": user_query}]}
            agent_response = await self.agent.ainvoke(agent_input)

            # Step 2: Extract messages and initialize decision
            messages = agent_response.get("messages", [])
            decision = None

            # Step 3: Search messages (reverse order = latest first)
            # Look for the first valid tool result to avoid unnecessary processing
            for msg in reversed(messages):
                # 3.1 If it's a ToolMessage (result from finalize_routing tool)
                if msg.__class__.__name__ == "ToolMessage":
                    decision = getattr(msg, "content", None)
                    # If we get a valid decision from tool, use it immediately
                    if decision and decision in [RoutingDecision.CLEAR_QUESTION.value, RoutingDecision.AMBIGUOUS.value]:
                        logger.info(f"[RouterAgent] Found valid tool result: {decision}")
                        return {
                            **state,
                            "routing_decision": decision,
                            "response": decision,
                        }

                # 3.2 If it's an AIMessage that triggered tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") == "finalize_routing":
                            args = tool_call.get("args", {})
                            decision = args.get("decision")
                            # If we have a valid decision from tool call, use it immediately
                            if decision and decision in [RoutingDecision.CLEAR_QUESTION.value, RoutingDecision.AMBIGUOUS.value]:
                                logger.info(f"[RouterAgent] Found valid tool call: {decision}")
                                return {
                                    **state,
                                    "routing_decision": decision,
                                    "response": decision,
                                }

            # Step 4: Fallback — try parsing last AI message text if no valid tool result found
            last_message = messages[-1] if messages else None
            if last_message and hasattr(last_message, "content"):
                decision = getattr(last_message, "content", "").strip()
                logger.warning(f"[RouterAgent] No valid tool call found, using last message content: {decision}")
            else:
                decision = RoutingDecision.AMBIGUOUS.value
                logger.warning(f"[RouterAgent] No decision found, using fallback: {decision}")

            # Step 5: Validate and normalize decision
            if decision:
                decision = str(decision).strip().lower()
                logger.info(f"[RouterAgent] Extracted decision: '{decision}'")

            if decision not in [
                RoutingDecision.CLEAR_QUESTION.value,
                RoutingDecision.AMBIGUOUS.value,
            ]:
                logger.warning(f"[RouterAgent] Unexpected output: {decision} → fallback to 'ambiguous'")
                decision = RoutingDecision.AMBIGUOUS.value

            # Step 6: Return updated state
            return {
                **state,
                "routing_decision": decision,
                "response": decision,
            }

        except Exception as e:
            logger.error(f"[RouterAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"RouterAgent error: [{type(e).__name__}]: {str(e)}",
            )
