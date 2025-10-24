from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from domain.enums.llm_type import LLMType
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from typing import Dict, Any
import asyncio

class ToolExecutor:
    """
    Tool Executor that dynamically executes tools based on the planner's decision.

    Supports different execution strategies:
    - Single tool execution (RAG or Web)
    - Parallel execution (Hybrid)
    - Conditional execution (RAG then Web)
    """

    def __init__(self, agents: Dict[str, Any]):
        """
        Initialize the tool executor with access to all agent instances.

        Args:
            agents: Dictionary of agent instances from WorkflowGraph
        """
        self.agents = agents

    async def _execute_rag_search(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute RAG search tool."""
        logger.info("[ToolExecutor] Executing RAG search")

        try:
            # Execute RAG retrieval
            rag_state = await self.agents[LLMType.RAG_RETRIEVAL_AGENT.value].invoke(state)

            # Evaluate confidence
            eval_state = await self.agents[LLMType.CONFIDENCE_EVALUATOR_AGENT.value].invoke(rag_state)

            return {
                "success": True,
                "documents": eval_state.get("retrieved_documents", []),
                "scores": eval_state.get("retrieval_scores", []),
                "confidence": eval_state.get("confidence_score", 0.0),
                "needs_more_info": eval_state.get("needs_web_search", False)
            }
        except Exception as e:
            logger.error(f"[ToolExecutor] RAG search failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "confidence": 0.0
            }

    async def _execute_web_search(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute web search tool."""
        logger.info("[ToolExecutor] Executing web search")

        try:
            web_state = await self.agents[LLMType.WEB_SEARCH_AGENT.value].invoke(state)

            return {
                "success": True,
                "results": web_state.get("web_search_results", []),
                "count": len(web_state.get("web_search_results", []))
            }
        except Exception as e:
            logger.error(f"[ToolExecutor] Web search failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    async def _execute_hybrid_search(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute RAG and web search in parallel."""
        logger.info("[ToolExecutor] Executing hybrid search (parallel RAG + Web)")

        # Execute both searches in parallel
        rag_task = self._execute_rag_search(state)
        web_task = self._execute_web_search(state)

        rag_result, web_result = await asyncio.gather(rag_task, web_task)

        return {
            "rag": rag_result,
            "web": web_result,
            "strategy": "parallel"
        }

    async def _execute_rag_then_web(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute RAG first, then web search if RAG confidence is low."""
        logger.info("[ToolExecutor] Executing RAG-then-Web (conditional)")

        # First, try RAG
        rag_result = await self._execute_rag_search(state)

        result = {
            "rag": rag_result,
            "web": None,
            "strategy": "rag_only"
        }

        # Check if we need web search
        needs_web = rag_result.get("needs_more_info", False)

        if needs_web and rag_result.get("success", False):
            logger.info("[ToolExecutor] RAG confidence low, executing web search")

            # Update state with RAG results before web search
            updated_state = {
                **state,
                "retrieved_documents": rag_result.get("documents", []),
                "retrieval_scores": rag_result.get("scores", []),
                "confidence_score": rag_result.get("confidence", 0.0),
                "needs_web_search": True
            }

            web_result = await self._execute_web_search(updated_state)
            result["web"] = web_result
            result["strategy"] = "rag_then_web"
        elif not rag_result.get("success", False):
            # RAG failed, try web search as fallback
            logger.warning("[ToolExecutor] RAG failed, falling back to web search")
            web_result = await self._execute_web_search(state)
            result["web"] = web_result
            result["strategy"] = "web_fallback"

        return result

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the planned tools and update the state with results.

        Args:
            state: Current workflow state with planned_tools

        Returns:
            Updated workflow state with tool_results
        """
        try:
            planned_tools = state.get("planned_tools", [])

            if not planned_tools:
                logger.warning("[ToolExecutor] No tools planned, skipping execution")
                return {
                    **state,
                    "tool_results": {"message": "No tools were planned"}
                }

            logger.info(f"[ToolExecutor] Executing planned tools: {planned_tools}")

            # Determine execution strategy
            tool_results = {}

            # Handle "none" case (no search needed)
            if "none" in planned_tools:
                logger.info("[ToolExecutor] No search tools needed")
                tool_results = {"strategy": "none", "message": "No search executed"}

            # Handle hybrid search
            elif "hybrid_search" in planned_tools:
                tool_results = await self._execute_hybrid_search(state)

            # Handle RAG then Web
            elif "rag_then_web" in planned_tools:
                tool_results = await self._execute_rag_then_web(state)

            # Handle multiple tools explicitly listed
            elif len(planned_tools) > 1:
                # Execute in parallel
                tasks = []
                tool_names = []

                if "rag_search" in planned_tools:
                    tasks.append(self._execute_rag_search(state))
                    tool_names.append("rag")

                if "web_search" in planned_tools:
                    tasks.append(self._execute_web_search(state))
                    tool_names.append("web")

                results = await asyncio.gather(*tasks)
                tool_results = {
                    name: result for name, result in zip(tool_names, results)
                }
                tool_results["strategy"] = "parallel"

            # Handle single tool
            elif "rag_search" in planned_tools:
                rag_result = await self._execute_rag_search(state)
                tool_results = {"rag": rag_result, "strategy": "rag_only"}

            elif "web_search" in planned_tools:
                web_result = await self._execute_web_search(state)
                tool_results = {"web": web_result, "strategy": "web_only"}

            else:
                logger.warning(f"[ToolExecutor] Unknown tools: {planned_tools}")
                tool_results = {"error": f"Unknown tools: {planned_tools}"}

            # Update state with all relevant information
            updated_state = {
                **state,
                "tool_results": tool_results
            }

            # Extract and add specific fields for backward compatibility
            if "rag" in tool_results and tool_results["rag"].get("success"):
                rag_data = tool_results["rag"]
                updated_state.update({
                    "retrieved_documents": rag_data.get("documents", []),
                    "retrieval_scores": rag_data.get("scores", []),
                    "confidence_score": rag_data.get("confidence", 0.0),
                    "needs_web_search": rag_data.get("needs_more_info", False)
                })

            if "web" in tool_results and tool_results["web"] and tool_results["web"].get("success"):
                web_data = tool_results["web"]
                updated_state["web_search_results"] = web_data.get("results", [])

            logger.info(f"[ToolExecutor] Execution complete. Strategy: {tool_results.get('strategy', 'unknown')}")

            return updated_state

        except Exception as e:
            logger.error(f"[ToolExecutor] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"ToolExecutor error: [{type(e).__name__}]: {str(e)}",
            )
