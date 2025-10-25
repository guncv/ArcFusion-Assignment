from pydantic import BaseModel, Field
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
            orchestration_attempts = state.get("orchestration_attempts", 0)
            orchestration_history = state.get("orchestration_history", [])

            response = await self.chain.ainvoke({
                "user_query": user_query,
                "generated_answer": generated_answer,
            })

            is_sufficient = response.get("is_sufficient", False)
            issues = response.get("issues", "No issues found")

            # Add reflection result to history
            if orchestration_history:
                orchestration_history[-1]["reflection_result"] = {
                    "is_sufficient": is_sufficient,
                    "issues": issues,
                    "answer_length": len(generated_answer),
                }

            # Increment orchestration_attempts if answer is insufficient
            # This must be done here (not in routing function) for state to persist
            new_attempts = orchestration_attempts + 1 if not is_sufficient else orchestration_attempts

            return {
                **state,
                "is_answer_sufficient": is_sufficient,
                "reflection_issues": issues,
                "orchestration_attempts": new_attempts,
                "orchestration_history": orchestration_history,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"Orchestration Reflection Agent error: [{type(e).__name__}]: {str(e)}",
            )
