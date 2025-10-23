from langgraph.graph import END
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from infrastructure.llm.loader import loadLLM, getChatHistory
from domain.enums.llm_type import LLMType
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from langchain_core.output_parsers import StrOutputParser
from prompts.refined_agent_prompt import REFINED_QUERY_AGENT_PROMPT

class RefinedQueryAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.REFINED_QUERY_AGENT)
        self.chain = REFINED_QUERY_AGENT_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            session_id = state.get("session_id", "")

            chat_history = getChatHistory(session_id)
            response = await self.chain.ainvoke({
                "user_query": user_query,
                "history": chat_history.messages
            })
            
            return {
                **state,
                "user_query": response,
            }
        except Exception as e:
            logger.error(f"[ClarificationAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"ClarificationAgent error: [{type(e).__name__}]: {str(e)}",
            )
