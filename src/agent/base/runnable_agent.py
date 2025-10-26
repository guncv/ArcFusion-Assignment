from typing import Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import BaseOutputParser
from src.graph import WorkflowState
from src.constants import LLMType
from src.infras.llm import loadLLM, getChatHistory
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from .agent_interface import AgentInterface

class RunnableAgent(AgentInterface):

    def __init__(
        self,
        llm_type: LLMType,
        prompt: ChatPromptTemplate,
        parser: BaseOutputParser,
        agent_name: Optional[str] = None
    ):
        self.llm_type = llm_type
        self.llm = loadLLM(llm_type)
        self.prompt = prompt
        self.parser = parser
        self._agent_name = agent_name

        self.chain = self.prompt | self.llm | self.parser

    @property
    def name(self) -> str:
        # Get the agent's name.
        return self._agent_name or super().name

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        # Process the workflow state using the LCEL chain.
        try:
            # Subclasses should override this with specific implementation
            raise NotImplementedError(
                f"{self.name} must implement the ainvoke method"
            )
        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

    def get_chat_history(self, state: WorkflowState):
        # Helper method to get chat history from state.
        session_id = state.get("session_id", "")
        return getChatHistory(session_id)

    async def ainvoke_chain(self, inputs: dict) -> Any:
        # Invoke the LCEL chain with the given inputs.
        return await self.chain.ainvoke(inputs)

runnable_agent = RunnableAgent()