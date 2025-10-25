from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from prompts.planner_agent_prompt import PLANNER_AGENT_PROMPT
from prompts.planner_context_templates import (
    NO_RAG_CONTEXT_TEMPLATE,
    INITIAL_PLANNING_CONTEXT_TEMPLATE,
    REPLANNING_CONTEXT_TEMPLATE
)
from infrastructure.llm.loader import loadLLM
from domain.enums.llm_type import LLMType
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser

class SearchQuery(BaseModel):
    query: str = Field(description="The specific search query to execute")
    purpose: str = Field(description="What information this query aims to find")

class ExecutionPlan(BaseModel):
    search_queries: List[SearchQuery] = Field(
        description="List of specific search queries to execute. Generate 1-3 targeted queries to find the missing information. Each query will be executed by a separate worker in parallel."
    )

class PlannerAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.PLANNER_AGENT)
        self.chain = PLANNER_AGENT_PROMPT | self.llm | JsonOutputParser(pydantic_object=ExecutionPlan)

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            is_answer_sufficient = state.get("is_answer_sufficient", False)
            rag_synthesizer_response = state.get("rag_synthesizer_response", "")
            rag_reflection_comment = state.get("rag_reflection_comment", "")
            
            # Determine context based on workflow state
            if not is_answer_sufficient:
                reflection_issues = state.get("reflection_issues", "")
                old_response = state.get("response", "")
                old_queries = state.get("old_queries", [])
                
                context = REPLANNING_CONTEXT_TEMPLATE.format(
                    user_query=user_query,
                    old_response=old_response,
                    reflection_comment=reflection_issues,
                    old_queries=old_queries
                )
            elif rag_synthesizer_response or rag_reflection_comment:
                context = INITIAL_PLANNING_CONTEXT_TEMPLATE.format(
                    user_query=user_query,
                    rag_reflection_comment=rag_reflection_comment
                )
            else:
                context = NO_RAG_CONTEXT_TEMPLATE.format(
                    user_query=user_query,
                )

            execution_plan = await self.chain.ainvoke({
                "context": context,
            })
            
            existing_old_queries = state.get("old_queries", [])
            new_queries = execution_plan.get("search_queries", [])
            
            return {
                **state,
                "generated_queries": new_queries,
                "old_queries": existing_old_queries + new_queries,
            }

        except Exception as e:
            logger.error(f"[PlannerAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"PlannerAgent error: [{type(e).__name__}]: {str(e)}",
            )
