from pydantic import BaseModel, Field
from src.graph import WorkflowState
from src.constants import LLMAgentName, LLMType
from src.prompts import REFLECTION_PROMPT
from langchain_core.output_parsers import JsonOutputParser
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import runnable_agent

class ReflectionResult(BaseModel):
    is_sufficient: bool = Field(description="Whether the answer is sufficient and complete")
    issues: str = Field(description="A string describing the issues with the answer if it is insufficient and what should be done to improve it for planner to fix it")

class ReflectionAgent(runnable_agent):
    def __init__(self):
        parser = JsonOutputParser(pydantic_object=ReflectionResult)
        super().__init__(
            llm_type=LLMType.REFLECTION_AGENT,
            prompt=REFLECTION_PROMPT,
            parser=parser,
            agent_name=LLMAgentName.REFLECTION_AGENT.value
        )

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            generated_answer = state.get("response", "")
            orchestration_attempts = state.get("orchestration_attempts", 0)
            orchestration_history = state.get("orchestration_history", [])

            response = await self.ainvoke_chain({
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
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

reflection_agent = ReflectionAgent()