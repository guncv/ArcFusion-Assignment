from src.graph import WorkflowState, ToolType
from src.constants import LLMAgentName
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import AgentInterface
import asyncio
from src.agent.autonomous.web_search_agent import WebSearchAgent
from src.agent.autonomous.rag_retrieval_agent import RAGRetrievalAgent
from src.infras.log import logger

class ToolExecutor(AgentInterface):
    def __init__(self):
        self.web_search_agent = WebSearchAgent()
        self.rag_retrieval_agent = RAGRetrievalAgent()
        self._agent_name = LLMAgentName.TOOL_EXECUTOR.value

    @property
    def name(self) -> str:
        return self._agent_name or super().name

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            selected_tool = state.get("selected_tool", ToolType.WEB_SEARCH.value)
            generated_queries = state.get("generated_queries", [])
            session_id = state.get("session_id", "unknown")

            logger.info(f"[{session_id}] ToolExecutor: tool={selected_tool}, queries={len(generated_queries)}")

            if selected_tool == ToolType.NONE.value:
                logger.debug(f"[{session_id}] No tool selected, skipping search")
                return {
                    **state,
                    "web_search_results": [],
                    "retrieved_documents_with_scores": [],
                }

            if not generated_queries:
                user_query = state.get("user_query", "")
                generated_queries = [{"query": user_query, "purpose": "Answer user question"}]
                logger.debug(f"[{session_id}] No queries generated, using user query as fallback")

            worker_tasks = []
            for i, query_obj in enumerate(generated_queries, 1):
                query_text = query_obj.get("query", "")
                query_purpose = query_obj.get("purpose", "")
                logger.info(f"[{session_id}]   Executing Query {i}/{len(generated_queries)}: '{query_text}' | Purpose: {query_purpose}")

                if selected_tool == ToolType.RAG_SEARCH.value:
                    task = self.rag_retrieval_agent.search(query_text, query_purpose)
                else:
                    task = self.web_search_agent.ainvoke(query_text, query_purpose)

                worker_tasks.append(task)

            logger.info(f"[{session_id}] Executing {len(worker_tasks)} parallel {selected_tool} tasks")
            raw_results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            search_results = []
            successful_workers = 0
            failed_workers = 0

            for i, result in enumerate(raw_results, 1):
                if isinstance(result, Exception):
                    logger.warning(f"[{session_id}]   Worker {i} failed: {str(result)}")
                    failed_workers += 1
                    continue

                if isinstance(result, list):
                    result_count = len(result)
                    search_results.extend(result)
                    successful_workers += 1
                    logger.info(f"[{session_id}]   Worker {i} succeeded: returned {result_count} results")
                else:
                    search_results.append(result)
                    successful_workers += 1
                    logger.info(f"[{session_id}]   Worker {i} succeeded: returned 1 result")

            logger.info(f"[{session_id}] ToolExecutor completed: {successful_workers} succeeded, {failed_workers} failed, {len(search_results)} total results")

            if selected_tool == ToolType.RAG_SEARCH.value:
                return {
                    **state,
                    "retrieved_documents_with_scores": search_results,
                }
            else:
                return {
                    **state,
                    "web_search_results": search_results,
                }

        except Exception as e:
            logger.error(f"ToolExecutor error: {str(e)}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )