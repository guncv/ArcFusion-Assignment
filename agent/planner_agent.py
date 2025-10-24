from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from prompts.planner_agent_prompt import PLANNER_AGENT_PROMPT
from prompts.planner_context_templates import (
    INITIAL_PLANNING_CONTEXT_TEMPLATE,
    REPLANNING_CONTEXT_TEMPLATE
)
from infrastructure.llm.loader import loadLLM
from domain.enums.llm_type import LLMType
from langchain_core.tools import tool
from langchain.agents import create_agent
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from pydantic import BaseModel, Field
from typing import List

class ExecutionPlanInput(BaseModel):
    tools: List[str] = Field(description="List of tools to execute: ['rag_search', 'web_search', 'hybrid_search', 'rag_then_web', 'none']")
    reasoning: str = Field(description="Explanation for why these tools were chosen")

@tool(args_schema=ExecutionPlanInput)
def finalize_execution_plan(tools: List[str], reasoning: str) -> dict:
    """
    Finalizes the execution plan with the list of tools to use.
    Use this tool to submit your final execution plan.

    Args:
        tools: List of tools to execute (e.g., ['rag_search'], ['web_search'], ['rag_search', 'web_search'])
        reasoning: Explanation for why these tools were chosen

    Returns:
        dict: The finalized execution plan
    """
    valid_tools = [
        "web_search",
        "hybrid_search",
        "rag_then_web",
        "none"
    ]

    # Validate tools
    cleaned_tools = [tool.strip().lower() for tool in tools]
    invalid_tools = [t for t in cleaned_tools if t not in valid_tools]

    if invalid_tools:
        logger.error(f"[Tool - finalize_execution_plan] Invalid tools: {invalid_tools}")
        return {
            "status": "error",
            "tools": [],
            "reasoning": f"Invalid tools specified: {invalid_tools}"
        }

    logger.info(f"[Tool - finalize_execution_plan] Plan: {cleaned_tools} | Reasoning: {reasoning}")

    return {
        "status": "success",
        "tools": cleaned_tools,
        "reasoning": reasoning
    }

class PlannerAgent:
    """
    Planner Agent that dynamically decides which tools to use based on the user query.

    This agent analyzes the query and determines the optimal execution strategy:
    - RAG only: For questions clearly answerable from knowledge base
    - Web search only: For current events, real-time data, or topics not in KB
    - Hybrid: For complex queries needing both knowledge base and current info
    - RAG then Web: Try RAG first, fall back to web if insufficient
    """

    def __init__(self):
        self.llm = loadLLM(LLMType.PLANNER_AGENT)
        self.tools = [finalize_execution_plan]
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=PLANNER_AGENT_PROMPT
        )

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            refined_query = state.get("refined_query", user_query)
            is_replanning = state.get("is_replanning", False)
            previous_attempts = state.get("previous_attempts", [])

            # Build context based on whether this is initial or replanning
            if is_replanning and previous_attempts:
                # Replanning context with feedback from previous attempt
                last_attempt = previous_attempts[-1]
                previous_tools = last_attempt.get("tools", [])
                previous_confidence = last_attempt.get("confidence", 0.0)
                tool_results = last_attempt.get("tool_results", {})

                context = REPLANNING_CONTEXT_TEMPLATE.format(
                    user_query=user_query,
                    refined_query=refined_query,
                    previous_tools=previous_tools,
                    previous_confidence=previous_confidence,
                    strategy=tool_results.get('strategy', 'unknown')
                )
            else:
                # Initial planning context
                context = INITIAL_PLANNING_CONTEXT_TEMPLATE.format(
                    user_query=user_query,
                    refined_query=refined_query
                )

            # Run the agent
            agent_input = {"messages": [{"role": "user", "content": context}]}
            agent_response = await self.agent.ainvoke(agent_input)

            # Extract the execution plan
            messages = agent_response.get("messages", [])
            execution_plan = None

            # Search for tool results (reverse order = latest first)
            for msg in reversed(messages):
                if msg.__class__.__name__ == "ToolMessage":
                    content = getattr(msg, "content", None)
                    if content:
                        try:
                            # Content might be a string representation of dict
                            if isinstance(content, str):
                                import ast
                                execution_plan = ast.literal_eval(content)
                            else:
                                execution_plan = content

                            if execution_plan.get("status") == "success":
                                logger.info(f"[PlannerAgent] Found valid plan: {execution_plan}")
                                break
                        except Exception as e:
                            logger.warning(f"[PlannerAgent] Failed to parse tool result: {e}")

                # Check tool calls in AIMessage
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") == "finalize_execution_plan":
                            args = tool_call.get("args", {})
                            execution_plan = {
                                "status": "success",
                                "tools": args.get("tools", []),
                                "reasoning": args.get("reasoning", "")
                            }
                            logger.info(f"[PlannerAgent] Found valid tool call: {execution_plan}")
                            break

            # Fallback if no plan found
            if not execution_plan or execution_plan.get("status") != "success":
                logger.warning("[PlannerAgent] No valid plan found, using fallback: RAG then Web")
                execution_plan = {
                    "status": "success",
                    "tools": ["rag_then_web"],
                    "reasoning": "Fallback strategy: try RAG first, then web search if needed"
                }

            # Update state with the plan
            return {
                **state,
                "execution_plan": execution_plan["reasoning"],
                "planned_tools": execution_plan["tools"],
                "tool_results": {}
            }

        except Exception as e:
            logger.error(f"[PlannerAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"PlannerAgent error: [{type(e).__name__}]: {str(e)}",
            )
