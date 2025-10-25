from domain.enums.workflow_state import WorkflowState, RoutingDecision
from prompts.clarification_agent_prompt import CLARIFICATION_ROUTING_PROMPT
from infrastructure.llm.loader import getChatHistory, loadLLM
from domain.enums.llm_type import LLMType
from langchain_core.tools import tool
from langchain.agents import create_agent
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from pydantic import BaseModel, Field

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

class ClarificationAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.CLARIFICATION_AGENT)
        self.tools = [finalize_clarification_routing] 
        
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=CLARIFICATION_ROUTING_PROMPT
        )

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            session_id = state.get("session_id", "")
            
            # Get chat history for context
            chat_history = getChatHistory(session_id)
            history_messages = chat_history.messages
            
            # Build messages list with history + current query
            messages = []
            
            # Add chat history
            for msg in history_messages:
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.type == 'ai':
                        messages.append({"role": "assistant", "content": msg.content})
            
            # Add current user query
            messages.append({"role": "user", "content": user_query})
            
            response = await self.agent.ainvoke({"messages": messages})
            
            # Extract the routing decision from tool calls
            messages = response.get("messages", [])

            # Valid routing decisions
            valid_decisions = [
                RoutingDecision.SMALLTALK.value,
                RoutingDecision.NEEDS_MORE_DETAIL.value,
                RoutingDecision.PROCESS_QUERY.value,
            ]

            # Step 3: Search messages (reverse order = latest first)
            # Look for the first valid tool result to avoid unnecessary processing
            for msg in reversed(messages):
                # 3.1 If it's a ToolMessage (result from finalize_routing tool)
                if msg.__class__.__name__ == "ToolMessage":
                    decision = getattr(msg, "content", None)
                    # If we get a valid decision from tool, use it immediately
                    if decision and decision in valid_decisions:
                        return {
                            **state,
                            "routing_decision": decision,
                        }

                # 3.2 If it's an AIMessage that triggered tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") == "finalize_clarification_routing":
                            args = tool_call.get("args", {})
                            decision = args.get("decision")
                            # If we have a valid decision from tool call, use it immediately
                            if decision and decision in valid_decisions:
                                return {
                                    **state,
                                    "routing_decision": decision,
                                }

            # Step 4: Fallback — try parsing last AI message text if no valid tool result found
            last_message = messages[-1] if messages else None
            if last_message and hasattr(last_message, "content"):
                decision = getattr(last_message, "content", "").strip()
            else:
                decision = RoutingDecision.NEEDS_MORE_DETAIL.value
            # Step 5: Validate and normalize decision
            if decision:
                decision = str(decision).strip().lower()

            if decision not in valid_decisions:
                decision = RoutingDecision.NEEDS_MORE_DETAIL.value

            # Step 6: Return updated state
            return {
                **state,
                "routing_decision": decision,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"ClarificationAgent error: [{type(e).__name__}]: {str(e)}",
            )
