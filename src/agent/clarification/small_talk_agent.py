from src.graph import WorkflowState
from src.constants import LLMAgentName, LLMType
from langchain_core.output_parsers import StrOutputParser
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.prompts import SMALLTALK_PROMPT
from src.agent.base import runnable_agent

class SmallTalkAgent(runnable_agent):
    def __init__(self):
        super().__init__(
            llm_type=LLMType.SMALLTALK_AGENT,
            prompt=SMALLTALK_PROMPT,
            parser=StrOutputParser(),
            agent_name=LLMAgentName.SMALLTALK_AGENT.value
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

small_talk_agent = SmallTalkAgent()