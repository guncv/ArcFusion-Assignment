from domain.enums.workflow_state import WorkflowState, RoutingDecision
from prompts.intent_analysis_agent_prompt import INTENT_ANALYSIS_AGENT_PROMPT
from infrastructure.llm.loader import loadLLM
from domain.enums.llm_type import LLMType
from langchain_core.tools import tool
from langchain.agents import create_agent
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from pydantic import BaseModel, Field

class IntentDecisionInput(BaseModel):
    decision: str = Field(description="The intent decision: 'use_rag' or 'use_web_search'")

@tool(args_schema=IntentDecisionInput)
def finalize_intent_decision(decision: str) -> str:
    """
    Validates and finalizes the intent decision after LLM analysis.
    This tool determines whether to use RAG (documents) or web search (real-time data).
    Use this tool to submit your final intent decision.

    Args:
        decision: The final intent decision ('use_rag' or 'use_web_search').

    Returns:
        str: Confirmation of the validated decision.
    """
    cleaned_decision = decision.strip().lower()
    valid_decisions = [RoutingDecision.USE_RAG.value, RoutingDecision.USE_WEB_SEARCH.value]

    if cleaned_decision not in valid_decisions:
        return "invalid_decision"

    return cleaned_decision

class IntentAnalysisAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.INTENT_ANALYSIS_AGENT)
        self.tools = [finalize_intent_decision]
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=INTENT_ANALYSIS_AGENT_PROMPT
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
                # 3.1 If it's a ToolMessage (result from finalize_intent tool)
                if msg.__class__.__name__ == "ToolMessage":
                    decision = getattr(msg, "content", None)
                    # If we get a valid decision from tool, use it immediately
                    if decision and decision in [RoutingDecision.USE_RAG.value, RoutingDecision.USE_WEB_SEARCH.value]:
                        return {
                            **state,
                            "routing_decision": decision,
                        }

                # 3.2 If it's an AIMessage that triggered tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") == "finalize_intent_decision":
                            args = tool_call.get("args", {})
                            decision = args.get("decision")
                            # If we have a valid decision from tool call, use it immediately
                            if decision and decision in [RoutingDecision.USE_RAG.value, RoutingDecision.USE_WEB_SEARCH.value]:
                                return {
                                    **state,
                                    "routing_decision": decision,
                                }

            # Step 4: Fallback — try parsing last AI message text if no valid tool result found
            last_message = messages[-1] if messages else None
            if last_message and hasattr(last_message, "content"):
                decision = getattr(last_message, "content", "").strip()
            else:
                decision = RoutingDecision.USE_RAG.value  # Default to RAG if unclear
            # Step 5: Validate and normalize decision
            if decision:
                decision = str(decision).strip().lower()
            if decision not in [
                RoutingDecision.USE_RAG.value,
                RoutingDecision.USE_WEB_SEARCH.value,
            ]:
                decision = RoutingDecision.USE_RAG.value

            # Step 6: Return updated state
            return {
                **state,
                "routing_decision": decision,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"IntentAnalysisAgent error: [{type(e).__name__}]: {str(e)}",
            )
