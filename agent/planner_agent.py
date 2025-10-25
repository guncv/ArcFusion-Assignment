from domain.enums.workflow_state import WorkflowState, ToolType
from prompts.planner_agent_prompt import PLANNER_AGENT_PROMPT
from prompts.planner_context_templates import (
    INITIAL_PLANNING_TEMPLATE,
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
    tool: str = Field(
        description=f"The tool to use: '{ToolType.RAG_SEARCH.value}' for internal documents, '{ToolType.WEB_SEARCH.value}' for real-time/external data, '{ToolType.NONE.value}' if no search needed"
    )
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
            orchestration_attempts = state.get("orchestration_attempts", 0)

            # Initialize history tracking
            orchestration_history = state.get("orchestration_history", [])

            # Determine context based on workflow state
            # SCENARIO 1: Replanning
            if orchestration_attempts > 0 and not is_answer_sufficient:
                reflection_issues = state.get("reflection_issues", "")
                old_response = state.get("response", "")
                old_queries = state.get("old_queries", [])
                previous_tool = state.get("selected_tool", "unknown")

                context = REPLANNING_CONTEXT_TEMPLATE.format(
                    user_query=user_query,
                    previous_tool=previous_tool,
                    old_response=old_response,
                    reflection_comment=reflection_issues,
                    old_queries=old_queries
                )

            # SCENARIO 2: Initial planning (autonomous tool selection)
            else:
                context = INITIAL_PLANNING_TEMPLATE.format(
                    user_query=user_query,
                )

            execution_plan = await self.chain.ainvoke({
                "context": context,
            })

            # Extract tool and queries
            if isinstance(execution_plan, dict):
                selected_tool = execution_plan.get("tool", ToolType.WEB_SEARCH.value)
                new_queries = execution_plan.get("search_queries", [])
            else:
                # Fallback to default values
                selected_tool = ToolType.WEB_SEARCH.value
                new_queries = []

            # Track old queries for duplicate prevention
            existing_old_queries = state.get("old_queries", [])

            # Create detailed history entry for this iteration
            iteration_entry = {
                "iteration": orchestration_attempts,
                "is_replanning": orchestration_attempts > 0,
                "selected_tool": selected_tool,
                "generated_queries": new_queries,
                "context_type": "replanning" if orchestration_attempts > 0 else "initial_planning",
            }

            # Add replanning-specific details if applicable
            if orchestration_attempts > 0:
                iteration_entry["previous_tool"] = state.get("selected_tool", "unknown")
                iteration_entry["reflection_feedback"] = state.get("reflection_issues", "")
                iteration_entry["tool_switched"] = (selected_tool != state.get("selected_tool", "unknown"))

            # Add to history
            orchestration_history.append(iteration_entry)

            return {
                **state,
                "selected_tool": selected_tool,
                "generated_queries": new_queries,
                "old_queries": existing_old_queries + new_queries,
                "orchestration_history": orchestration_history,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"PlannerAgent error: [{type(e).__name__}]: {str(e)}",
            )
