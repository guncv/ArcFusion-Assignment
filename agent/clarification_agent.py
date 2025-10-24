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
    Validates and finalizes the clarification routing decision after LLM analysis.
    This tool only validates the decision format - the LLM makes the intelligent choice.
    Use this tool to submit your final routing decision.

    Args:
        decision: The final routing decision ('smalltalk', 'needs_clarification', or 'process_query').

    Returns:
        str: Confirmation of the validated decision.
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
        self.base_llm = loadLLM(LLMType.CLARIFICATION_AGENT)
        self.routing_tools = [finalize_clarification_routing]
        self.routing_agent = create_agent(
            model=self.base_llm,
            tools=self.routing_tools,
            system_prompt=CLARIFICATION_ROUTING_PROMPT
        )

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            session_id = state.get("session_id", "")
            
            # Get chat history for context
            chat_history = getChatHistory(session_id)
            logger.info(f"[ClarificationAgent] Chat history: {chat_history.messages}")
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
            
            response = await self.routing_agent.ainvoke({"messages": messages})
            
            # Extract the routing decision from tool calls
            messages = response.get("messages", [])
            routing_decision = None
            
            # Look for tool calls in the messages - take the FIRST valid decision
            valid_decisions = [
                RoutingDecision.SMALLTALK.value,
                RoutingDecision.NEEDS_MORE_DETAIL.value,
                RoutingDecision.PROCESS_QUERY.value
            ]
            
            for msg in messages:
                # Check for ToolMessage with the decision first (tool result)
                if hasattr(msg, "name") and msg.name == "finalize_clarification_routing":
                    decision = msg.content
                    if decision in valid_decisions:
                        logger.info(f"[ClarificationAgent] Found valid decision from tool message: {decision}")
                        return {
                            **state,
                            "routing_decision": decision,
                            "response": f"Routing decision: {decision}",
                        }
                
                # Check for tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") == "finalize_clarification_routing":
                            args = tool_call.get("args", {})
                            decision = args.get("decision")
                            if decision and decision in valid_decisions:
                                logger.info(f"[ClarificationAgent] Found valid decision from tool call: {decision}")
                                return {
                                    **state,
                                    "routing_decision": decision,
                                    "response": f"Routing decision: {decision}",
                                }
            
            # Fallback to last message content if no valid tool call found
            last_message = messages[-1] if messages else None
            if last_message and hasattr(last_message, "content"):
                routing_decision = last_message.content
                logger.warning(f"[ClarificationAgent] No valid tool call found, using last message content: {routing_decision}")
            else:
                routing_decision = RoutingDecision.PROCESS_QUERY.value
                logger.warning(f"[ClarificationAgent] No decision found, using fallback: {routing_decision}")
            
            logger.info(f"[ClarificationAgent] Routing decision: {routing_decision}")
            
            return {
                **state,
                "routing_decision": routing_decision,
            }

        except Exception as e:
            logger.error(f"[ClarificationAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"ClarificationAgent error: [{type(e).__name__}]: {str(e)}",
            )

