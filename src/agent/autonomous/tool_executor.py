from src.graph import WorkflowState, ToolType
from src.constants import LLMAgentName
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import agent_interface
import asyncio
from src.agent.autonomous.web_search_agent import WebSearchAgent
from src.agent.autonomous.rag_retrieval_agent import RAGRetrievalAgent

class ToolExecutor(agent_interface):
    def __init__(self):
        self.web_search_agent = WebSearchAgent()
        self.rag_retrieval_agent = RAGRetrievalAgent()
        self._agent_name = LLMAgentName.TOOL_EXECUTOR.value

    @property
    def name(self) -> str:
        return self._agent_name or super().name

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            # Get selected tool and queries from planner
            selected_tool = state.get("selected_tool", ToolType.WEB_SEARCH.value)
            generated_queries = state.get("generated_queries", [])
            
            # Handle 'none' tool - no search needed
            if selected_tool == ToolType.NONE.value:
                return {
                    **state,
                    "web_search_results": [],
                    "retrieved_documents_with_scores": [],
                }

            # Default to user query if no queries generated
            if not generated_queries:
                user_query = state.get("user_query", "")
                generated_queries = [{"query": user_query, "purpose": "Answer user question"}]

            # Spawn parallel workers based on selected tool
            worker_tasks = []
            for i, query_obj in enumerate(generated_queries, 1):
                query_text = query_obj.get("query", "")
                query_purpose = query_obj.get("purpose", "")

                # Route to appropriate agent
                if selected_tool == ToolType.RAG_SEARCH.value:
                    task = self.rag_retrieval_agent.search(query_text, query_purpose)
                else:  # web_search (default)
                    task = self.web_search_agent.search(query_text, query_purpose)

                worker_tasks.append(task)

            raw_results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            # Aggregate results from all workers
            search_results = []
            successful_workers = 0
            failed_workers = 0

            for i, result in enumerate(raw_results, 1):
                if isinstance(result, Exception):
                    failed_workers += 1
                    continue

                if isinstance(result, list):
                    search_results.extend(result)
                    successful_workers += 1
                else:
                    search_results.append(result)
                    successful_workers += 1

            # Store results in state based on tool type
            if selected_tool == ToolType.RAG_SEARCH.value:
                return {
                    **state,
                    "retrieved_documents_with_scores": search_results,
                }
            else:  # web_search
                return {
                    **state,
                    "web_search_results": search_results,
                }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

tool_executor = ToolExecutor()