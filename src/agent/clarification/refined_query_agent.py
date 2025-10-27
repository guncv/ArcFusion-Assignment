from src.graph import WorkflowState
from src.constants import LLMAgentName, LLMType
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.prompts import REFINED_QUERY_PROMPT
from src.agent.base import RunnableAgent
from src.infras.log import logger

class RefinedQueryAgent(RunnableAgent):
    def __init__(self):
        super().__init__(
            llm_type=LLMType.REFINED_QUERY_AGENT,
            prompt=REFINED_QUERY_PROMPT,
            parser=StrOutputParser(),
            agent_name=LLMAgentName.REFINED_QUERY_AGENT.value
        )

    def _convert_chat_messages_to_langchain(self, chat_messages):
        langchain_messages = []
        for msg in chat_messages:
            if msg.message_type == 'human':
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.message_type == 'ai':
                langchain_messages.append(AIMessage(content=msg.content))
        return langchain_messages

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            session_id = state.get("session_id", "unknown")
            chat_messages = await self.get_chat_history(state)

            logger.info(f"[{session_id}] RefinedQueryAgent refining query: '{user_query[:80]}...'")

            # Convert ChatMessage objects to LangChain message format
            history = self._convert_chat_messages_to_langchain(chat_messages)

            response = await self.ainvoke_chain({
                "user_query": user_query,
                "history": history
            })

            logger.info(f"[{session_id}] RefinedQueryAgent result: '{response[:80]}...'")

            return {
                **state,
                "user_query": response,
            }
        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )