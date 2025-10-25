from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
import asyncio
from agent.web_search_agent import WebSearchAgent

class ToolExecutor:
    def __init__(self):
        self.web_search_agent = WebSearchAgent()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            generated_queries = state.get("generated_queries", [])
            logger.info(f"[ToolExecutor] Generated queries: {generated_queries}")

            if not generated_queries:
                user_query = state.get("user_query", "")
                generated_queries = [{"query": user_query, "purpose": "Answer user question"}]

            num_workers = len(generated_queries)

            worker_tasks = []
            for i, query_obj in enumerate(generated_queries, 1):
                query_text = query_obj.get("query", "")
                query_purpose = query_obj.get("purpose", "")

                task = self.web_search_agent.invoke(query_text, query_purpose)
                worker_tasks.append(task)

            raw_results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            web_results = []
            successful_workers = 0
            failed_workers = 0

            for i, result in enumerate(raw_results, 1):
                if isinstance(result, Exception):
                    logger.error(f"[ToolExecutor] Worker-{i} FAILED: {result}")
                    failed_workers += 1
                    continue

                if isinstance(result, list):
                    web_results.extend(result)
                    successful_workers += 1
                    logger.info(f"[ToolExecutor] Worker-{i} SUCCESS: {len(result)} results")
                else:
                    web_results.append(result)
                    successful_workers += 1
                    logger.info(f"[ToolExecutor] Worker-{i} SUCCESS: 1 result")

            logger.info(
                f"[ToolExecutor] ✅ Parallel execution complete: "
                f"{successful_workers}/{num_workers} workers succeeded, "
                f"{len(web_results)} total results"
            )

            return {
                **state,
                "web_search_results": web_results,
            }

        except Exception as e:
            logger.error(f"[ToolExecutor] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"ToolExecutor error: [{type(e).__name__}]: {str(e)}",
            )
