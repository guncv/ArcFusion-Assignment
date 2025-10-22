from langgraph.graph import END
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState, RoutingDecision
from infrastructure.llm.loader import loadLLM, getChatHistory
from domain.enums.llm_type import LLMType
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent
from prompts.clarification_agent_prompt import CLARIFICATION_ROUTING_PROMPT
from pydantic import BaseModel, Field

class ClarificationRoutingInput(BaseModel):
    decision: str = Field(description="The routing decision: 'smalltalk', 'needs_clarification', or 'process_query'")

@tool(args_schema=ClarificationRoutingInput)
def finalize_clarification_routing(decision: str) -> str:
    """
    Finalizes the clarification routing decision after validation.
    Use this tool to submit your final routing decision.

    Args:
        decision: The final routing decision ('smalltalk', 'needs_clarification', or 'process_query').

    Returns:
        str: Confirmation of the finalized decision.
    """
    cleaned_decision = decision.strip().lower()
    valid_decisions = [
        RoutingDecision.SMALLTALK.value,
        RoutingDecision.NEEDS_MORE_DETAIL.value,
        RoutingDecision.PROCESS_QUERY.value
    ]

    if cleaned_decision not in valid_decisions:
        logger.error(f"[Tool - finalize_clarification_routing] Attempted to finalize invalid decision: {decision}")
        return "invalid_decision"

    logger.info(f"[Tool - finalize_clarification_routing] Finalized routing decision: {cleaned_decision}")
    return cleaned_decision

class ClarificationAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.CLARIFICATION_AGENT)

        self.routing_tools = [finalize_clarification_routing]
        self.routing_agent = create_agent(
            model=self.llm,
            tools=self.routing_tools,
            system_prompt=CLARIFICATION_ROUTING_PROMPT
        )

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            session_id = state.get("session_id", "")

            logger.info(f"[ClarificationAgent] Session ID: {session_id}")
            chat_history = getChatHistory(session_id)
            logger.info(f"[ClarificationAgent] Chat history message count: {len(chat_history.messages)}")

            # Format chat history for the routing agent (if any)
            history_str = ""
            if len(chat_history.messages) > 0:
                history_str = "\n".join([
                    f"{msg.type}: {msg.content}"
                    for msg in chat_history.messages[-5:]  # Last 5 messages for context
                ])
                logger.info(f"[ClarificationAgent] Including chat history in routing decision")
            else:
                logger.info(f"[ClarificationAgent] No chat history available")

            # Always use the routing agent to classify the query
            agent_input = {
                "messages": [{
                    "role": "user",
                    "content": f"Chat History:\n{history_str if history_str else '[]'}\n\nUser query: {user_query}"
                }]
            }

            agent_response = await self.routing_agent.ainvoke(agent_input)
            logger.info(f"[ClarificationAgent] Routing agent response: {agent_response}")

            # Extract decision from agent response
            messages = agent_response.get("messages", [])
            decision = None

            # Search for decision in tool messages
            for msg in reversed(messages):
                # Check ToolMessage
                if msg.__class__.__name__ == "ToolMessage":
                    decision = getattr(msg, "content", None)
                    if decision:
                        logger.info(f"[ClarificationAgent] Found decision in ToolMessage: {decision}")
                        break

                # Check AIMessage with tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") == "finalize_clarification_routing":
                            args = tool_call.get("args", {})
                            decision = args.get("decision")
                            if decision:
                                logger.info(f"[ClarificationAgent] Found decision in tool_call args: {decision}")
                                break
                    if decision:
                        break

            # Fallback to last message content
            if not decision:
                last_message = messages[-1] if messages else None
                if last_message and hasattr(last_message, "content"):
                    decision = getattr(last_message, "content", "").strip()
                    logger.warning(f"[ClarificationAgent] No tool call found, using last message content: {decision}")

            # Validate and normalize decision
            if decision:
                decision = str(decision).strip().lower()

            return {
                **state,
                "routing_decision": decision,
                "response": decision,
            }

        except Exception as e:
            logger.error(f"[ClarificationAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"ClarificationAgent error: [{type(e).__name__}]: {str(e)}",
            )
