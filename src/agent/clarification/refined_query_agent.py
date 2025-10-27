from src.graph import WorkflowState
from src.constants import LLMAgentName, LLMType
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.prompts import REFINED_QUERY_PROMPT
from src.agent.base import RunnableAgent

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
            chat_messages = await self.get_chat_history(state)
            
            # Convert ChatMessage objects to LangChain message format
            history = self._convert_chat_messages_to_langchain(chat_messages)
            
            response = await self.ainvoke_chain({
                "user_query": user_query,
                "history": history
            })
            
            return {
                **state,
                "user_query": response,
            }
        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

# Note: refined_query_agent should be instantiated with proper arguments when needed
# refined_query_agent = RefinedQueryAgent()