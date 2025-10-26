from src.graph import WorkflowState
from src.constants import LLMAgentName, LLMType
from langchain_core.output_parsers import StrOutputParser
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.prompts import REFINED_QUERY_PROMPT
from src.agent.base import runnable_agent

class RefinedQueryAgent(runnable_agent):
    def __init__(self):
        super().__init__(
            llm_type=LLMType.REFINED_QUERY_AGENT,
            prompt=REFINED_QUERY_PROMPT,
            parser=StrOutputParser(),
            agent_name=LLMAgentName.REFINED_QUERY_AGENT.value
        )

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            chat_history = self.get_chat_history(state)
            
            response = await self.ainvoke_chain({
                "user_query": user_query,
                "history": chat_history.messages
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

refined_query_agent = RefinedQueryAgent()