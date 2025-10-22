from langgraph.graph import END
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from infrastructure.llm.loader import loadLLM, getChatHistory
from domain.enums.llm_type import LLMType
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from langchain_core.output_parsers import StrOutputParser
from prompts.refined_agent_prompt import REFINED_QUERY_AGENT_PROMPT

class RefinedAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.REFINED_QUERY_AGENT)
        self.chain = REFINED_QUERY_AGENT_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")

            logger.info(f"[ClarificationAgent] Session ID: {state.get('session_id', '')}")
            chat_history = getChatHistory(state.get("session_id", ""))
            logger.info(f"[ClarificationAgent] Chat history: {chat_history.messages}")
            has_history = len(chat_history.messages) > 0

            logger.info(f"[ClarificationAgent] Session {state.get('session_id', '')} has history: {has_history}")
            logger.info(f"[ClarificationAgent] History message count: {len(chat_history.messages)}")

            if not has_history:
                response = await self.chain.ainvoke(user_query)
                logger.info(f"[ClarificationAgent] User query: {user_query}")
                logger.info(f"[ClarificationAgent] Agent response: {response}")
                return {
                    **state,
                    "response": response,
                    "routing_decision": END,
                }
            else:
                response = "This clarification agent should only be used when there's no conversation history."
                logger.warning(f"[ClarificationAgent] Agent invoked with existing history - this should not happen")
                return {
                    **state,
                    "response": "This clarification agent should only be used when there's no conversation history.",
                    "routing_decision": END,
                }

        except Exception as e:
            logger.error(f"[ClarificationAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"ClarificationAgent error: [{type(e).__name__}]: {str(e)}",
            )
