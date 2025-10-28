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
from typing import Optional
from langchain_core.output_parsers import JsonOutputParser
from src.agent.base import RunnableAgent
from src.infras.log import logger

class ExecutionPlan(BaseModel):
    reasoning: str = Field(
        description="Your autonomous reasoning: Why you chose this tool and queries. Explain your decision-making process."
    )
    tool: str = Field(
        description=f"The tool to use: '{ToolType.RAG_SEARCH.value}', '{ToolType.WEB_SEARCH.value}', or '{ToolType.HYBRID_SEARCH.value}'"
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query for rag_search or web_search (single tool only)"
    )
    rag_query: Optional[str] = Field(
        default=None,
        description="RAG-optimized query for hybrid_search (technical, paper-focused keywords)"
    )
    web_query: Optional[str] = Field(
        default=None,
        description="Web-optimized query for hybrid_search (biographical, external info keywords)"
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
            autonomous_attempts = state.get("autonomous_attempts", 0)

            # Replanning (autonomous rethinking based on reflection feedback)
            if autonomous_attempts > 0:
                comment = state.get("comment", "")
                current_response = state.get("response", "")
                previous_tool = state.get("selected_tool", "unknown")

                context = REPLANNING_CONTEXT_TEMPLATE.format(
                    user_query=user_query,
                    previous_tool=previous_tool,
                    current_response=current_response,
                    comment=comment,
                    attempt_number=autonomous_attempts,
                )
            # Initial planning (autonomous tool selection)
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
                query = execution_plan.get("query", "")
                rag_query = execution_plan.get("rag_query", "")
                web_query = execution_plan.get("web_query", "")
            else:
                selected_tool = ToolType.WEB_SEARCH.value
                query = ""
                rag_query = ""
                web_query = ""

            logger.info(f"PlannerAgent decision: tool={selected_tool}")

            return {
                **state,
                "selected_tool": selected_tool,
                "generated_queries": query,  # Keep for backward compatibility
                "rag_query": rag_query,
                "web_query": web_query,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )