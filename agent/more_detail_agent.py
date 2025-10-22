from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from infrastructure.llm.loader import loadLLM, getChatHistory
from domain.enums.llm_type import LLMType
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from langchain_core.output_parsers import StrOutputParser
from prompts.more_detail_agent_prompt import MORE_DETAIL_AGENT_PROMPT

class MoreDetailAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.NEEDS_MORE_DETAIL_AGENT)
        self.chain = MORE_DETAIL_AGENT_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            session_id = state.get("session_id", "")

            logger.info(f"[MoreDetailAgent] Session ID: {session_id}")
            chat_history = getChatHistory(session_id)
            logger.info(f"[MoreDetailAgent] Chat history message count: {len(chat_history.messages)}")

            # Format chat history for the prompt
            history_str = ""
            if len(chat_history.messages) > 0:
                history_str = "\n".join([
                    f"{msg.type}: {msg.content}"
                    for msg in chat_history.messages[-5:]  # Last 5 messages for context
                ])
                logger.info(f"[MoreDetailAgent] Including chat history in clarification")
            else:
                logger.info(f"[MoreDetailAgent] No chat history available")
                history_str = "No previous conversation history."

            response = await self.chain.ainvoke({
                "user_query": user_query,
                "chat_history": history_str
            })

            logger.info(f"[MoreDetailAgent] User query: {user_query}")
            logger.info(f"[MoreDetailAgent] Agent response: {response}")

            return {
                **state,
                "response": response,
            }

        except Exception as e:
            logger.error(f"[MoreDetailAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"MoreDetailAgent error: [{type(e).__name__}]: {str(e)}",
            )