from pydantic import BaseModel, Field
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from prompts.reflection_agent_prompt import REFLECTION_AGENT_PROMPT
from langchain_core.output_parsers import JsonOutputParser
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes

class ReflectionResult(BaseModel):
    is_sufficient: bool = Field(description="Whether the answer is sufficient and complete")
    issues: str = Field(description="A string describing the issues with the answer if it is insufficient and what should be done to improve it for planner to fix it")

class OrchestrationReflectionAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.ORCHESTRATION_REFLECTION_AGENT)
        self.chain = REFLECTION_AGENT_PROMPT | self.llm | JsonOutputParser(pydantic_object=ReflectionResult)

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            generated_answer = state.get("response", "")
            
            response = await self.chain.ainvoke({
                "user_query": user_query,
                "generated_answer": generated_answer,
            })

            return {
                **state,
                "is_answer_sufficient": response.get("is_sufficient", False),
                "reflection_issues": response.get("issues", "No issues found"),
            }

        except Exception as e:
            logger.error(f"[OrchestrationReflectionAgent] Error during reflection: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"Orchestration Reflection Agent error: [{type(e).__name__}]: {str(e)}",
            )