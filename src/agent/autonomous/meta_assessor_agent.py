from pydantic import BaseModel, Field
from src.infras import logger
from src.graph import WorkflowState
from src.constants import LLMAgentName, LLMType
from src.prompts import META_ASSESSOR_PROMPT
from langchain_core.output_parsers import JsonOutputParser
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import RunnableAgent
from src.constants.session import CONFIDENCE_THRESHOLD

class MetaAssessment(BaseModel):
    confidence_score: float = Field(
        description="Quality score from 0.0 to 1.0 indicating how well the retrieved information can answer the user's question"
    )
    reasoning: str = Field(
        description="Explanation of the confidence score: Why can/can't the retrieved information answer the question? What's present or missing?"
    )

class MetaAssessorAgent(RunnableAgent):

    def __init__(self):
        parser = JsonOutputParser(pydantic_object=MetaAssessment)
        super().__init__(
            llm_type=LLMType.META_ASSESSOR_AGENT,
            prompt=META_ASSESSOR_PROMPT,
            parser=parser,
            agent_name=LLMAgentName.META_ASSESSOR_AGENT.value
        )

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            generated_answer = state.get("response", "")

            # Evaluate the synthesized answer
            response = await self.ainvoke_chain({
                "user_query": user_query,
                "generated_answer": generated_answer,
            })

            confidence_score = response.get("confidence_score", 0.0)
            reasoning = response.get("reasoning", "")

            confidence_score = max(0.0, min(1.0, float(confidence_score)))

            # Increment autonomous attempts
            autonomous_attempts = state.get("autonomous_attempts", 0)
            logger.info(f"MetaAssessorAgent confidence_score: {confidence_score:.2f}")
            
            # Handle answer quality evaluation
            if confidence_score < CONFIDENCE_THRESHOLD:
                is_done = False
                new_attempts = autonomous_attempts + 1
                # Reset retrieval results to ensure fresh retrieval on replanning
                reset_state = {
                    "retrieved_documents_with_scores": [],
                    "web_search_results": [],
                }
            else:
                is_done = True
                new_attempts = autonomous_attempts
                reset_state = {}

            return {
                **state,
                **reset_state,
                "is_done": is_done,
                "comment": reasoning,
                "autonomous_attempts": new_attempts,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )