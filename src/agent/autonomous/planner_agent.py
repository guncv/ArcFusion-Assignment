from src.graph import WorkflowState, ToolType
from src.prompts import (
    PLANNER_PROMPT,
    INITIAL_PLANNING_TEMPLATE,
    REPLANNING_CONTEXT_TEMPLATE
)
from src.constants import LLMType, LLMAgentName
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from src.agent.base import RunnableAgent
from src.infras.log import logger

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

class PlannerAgent(RunnableAgent):
    def __init__(self):
        parser = JsonOutputParser(pydantic_object=ExecutionPlan)
        super().__init__(
            llm_type=LLMType.PLANNER_AGENT,
            prompt=PLANNER_PROMPT,
            parser=parser,
            agent_name=LLMAgentName.PLANNER_AGENT.value
        )

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
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

            execution_plan = await self.ainvoke_chain({
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

            session_id = state.get("session_id", "unknown")
            logger.info(f"[{session_id}] PlannerAgent generated {len(new_queries)} queries for tool={selected_tool}")
            for i, query_obj in enumerate(new_queries, 1):
                q = query_obj.get("query", "") if isinstance(query_obj, dict) else str(query_obj)
                p = query_obj.get("purpose", "N/A") if isinstance(query_obj, dict) else "N/A"
                logger.info(f"[{session_id}]   Query {i}: '{q}' | Purpose: {p}")

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
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

# Note: planner_agent should be instantiated with proper arguments when needed
# planner_agent = PlannerAgent()