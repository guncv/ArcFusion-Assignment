from src.graph import WorkflowState
from src.constants import LLMAgentName, LLMType
from langchain_core.messages import HumanMessage
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.prompts.clarification import REFINED_QUERY_SYSTEM_PROMPT
from src.agent.base import ReActAgent
from src.infras.log import logger
from langchain_core.tools import tool
from datetime import datetime, timezone

@tool
def get_current_datetime() -> str:
    """
    Get the current date and time. Use this when you need to resolve temporal references 
    like 'today', 'this month', 'last week', 'currently', etc.
    
    Args:
        timezone_offset: UTC timezone offset in hours (default: 0 for UTC)
        
    Returns:
        A string with current date and time information including:
        - Current date (YYYY-MM-DD)
        - Current time (HH:MM:SS)
        - Day of week
        - Month name
        - Year
        - ISO week number
        
    Examples:
        - User says "this month" → call this to know it's January 2025
        - User says "last week" → call this to calculate the date range
        - User says "today" → call this to get today's date
    """
    try:
        now = datetime.now(timezone.utc)
        
        return f"""Current Date & Time Information:
        - Date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})
        - Time: {now.strftime('%H:%M:%S')} UTC
        - Month: {now.strftime('%B %Y')}
        - Year: {now.year}
        - Week: Week {now.isocalendar()[1]} of {now.year}
        - Quarter: Q{(now.month - 1) // 3 + 1}

        Use this information to resolve temporal references in the user's query."""
    except Exception as e:
        return f"Error getting current datetime: {str(e)}"


class RefinedQueryAgent(ReActAgent):
    def __init__(self):
        super().__init__(
            llm_type=LLMType.REFINED_QUERY_AGENT,
            system_prompt=REFINED_QUERY_SYSTEM_PROMPT,
            tools=[get_current_datetime],
            agent_name=LLMAgentName.REFINED_QUERY_AGENT.value
        )

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            chat_messages = await self.get_chat_history(state)

            logger.info(f"RefinedQueryAgent refining query: '{user_query[:80]}...'")

            # Convert ChatMessage objects to LangChain message format
            history = self._convert_chat_messages_to_langchain(chat_messages)

            # Build the input for the ReAct agent
            input_messages = history + [
                HumanMessage(content=f"User query: {user_query}\n\nRefine this query based on the conversation history to make it more specific and clear question.")
            ]

            # Invoke the ReAct agent
            result = await self.agent.ainvoke({
                "messages": input_messages
            })

            # Extract the final answer from agent response
            if isinstance(result, dict) and "messages" in result:
                # Get the last AI message
                for msg in reversed(result["messages"]):
                    if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                        refined_query = msg.content.strip()
                        break
                else:
                    refined_query = user_query
            else:
                refined_query = user_query

            logger.info(f"RefinedQueryAgent result: '{refined_query[:80]}...'")

            return {
                **state,
                "user_query": refined_query,
            }
        except Exception as e:
            logger.error(f"RefinedQueryAgent error: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )