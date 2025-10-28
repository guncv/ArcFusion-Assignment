from pydantic import BaseModel, Field
from src.graph import WorkflowState
from src.constants import LLMAgentName, LLMType
from src.prompts import REFLECTION_PROMPT
from langchain_core.output_parsers import JsonOutputParser
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import RunnableAgent
from src.constants import CONFIDENCE_THRESHOLD

class ReflectionResult(BaseModel):
    is_done: bool = Field(
        description="Whether the answer is complete and addresses all aspects of the user's question"
    )
    reasoning: str = Field(
        description="Full autonomous analysis: Why this decision? What's the confidence/completeness? What's missing (if any)? What should the planner do next?"
    )

class ReflectionAgent(RunnableAgent):
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
            selected_tool = state.get("selected_tool", "")
            autonomous_attempts = state.get("autonomous_attempts", 0)

            confidence_score = state.get("confidence_score", 0.0)
            meta_assessment_reasoning = state.get("meta_assessment_reasoning", "")

            # Low confidence from MetaAssessor (no synthesized answer yet)
            if confidence_score < CONFIDENCE_THRESHOLD:
                is_done = False
                reasoning = (
                    f"MetaAssessor confidence is low ({confidence_score:.2f} < {CONFIDENCE_THRESHOLD}). "
                    f"The retrieved information isn't relevant to the user's question: {meta_assessment_reasoning}. "
                    f"Planner should replan with a different method or different query to retrieve more relevant information."
                )

            # High confidence from MetaAssessor (has synthesized answer)
            else:
                # Ask LLM to evaluate answer completeness
                response = await self.ainvoke_chain({
                    "user_query": user_query,
                    "generated_answer": generated_answer,
                    "selected_tool": selected_tool,
                    "confidence_score": confidence_score,
                })

                reasoning = response.get("reasoning", "")
                is_done = response.get("is_done", False)

            # Increment autonomous_attempts
            new_attempts = autonomous_attempts + 1

            return {
                **state,
                "is_done": is_done,
                "reflection_comment": reasoning,
                "autonomous_attempts": new_attempts,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )