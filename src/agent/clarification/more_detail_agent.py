from src.graph import WorkflowState
from src.constants import LLMAgentName, LLMType
from langchain_core.output_parsers import StrOutputParser
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.prompts import MORE_DETAIL_PROMPT
from src.agent.base import RunnableAgent

class MoreDetailAgent(RunnableAgent):
    def __init__(self):
        super().__init__(
            llm_type=LLMType.NEEDS_MORE_DETAIL_AGENT,
            prompt=MORE_DETAIL_PROMPT,
            parser=StrOutputParser(),
            agent_name=LLMAgentName.NEEDS_MORE_DETAIL_AGENT.value
        )

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            response = await self.ainvoke_chain({
                "input": user_query,
            })

            return {
                **state,
                "response": response,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

# Note: more_detail_agent should be instantiated with proper arguments when needed
# more_detail_agent = MoreDetailAgent()