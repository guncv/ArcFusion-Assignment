from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from infrastructure.llm.loader import loadLLM, getChatHistory
from domain.enums.llm_type import LLMType
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from langchain_core.output_parsers import StrOutputParser
from prompts.smalltalk_agent_prompt import SMALLTALK_AGENT_PROMPT

class SmallTalkAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.SMALLTALK_AGENT)
        self.chain = SMALLTALK_AGENT_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            session_id = state.get("session_id", "")

            chat_history = getChatHistory(session_id)
            response = await self.chain.ainvoke({
                "input": user_query,
                "history": chat_history.messages
            })

            return {
                **state,
                "response": response,
            }

        except Exception as e:
            logger.error(f"[SmallTalkAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"SmallTalkAgent error: [{type(e).__name__}]: {str(e)}",
            )