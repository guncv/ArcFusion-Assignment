import asyncio
from typing import Dict, Any, Optional
from src.graph.state import WorkflowState, ToolType
from src.agent import RAGEvaluator, WebEvaluator
from src.repositories.evaluation import get_evaluation_repository
from src.infras.log import logger

class EvaluationService:

    def __init__(self):
        self.rag_evaluator = RAGEvaluator()
        self.web_evaluator = WebEvaluator()
        self.evaluation_repo = get_evaluation_repository()

    async def evaluate_and_save(self, state: WorkflowState) -> None:
        try:
            selected_tool = state.get("selected_tool", "")
            current_response = state.get("current_synthesized_response", "")
            session_id = state.get("session_id", "unknown")
            user_query = state.get("user_query", "")

            if not current_response:
                logger.warning("No current_response to evaluate, skipping")
                return

            if selected_tool == ToolType.RAG_SEARCH.value:
                metrics = await self._evaluate_rag(state)
            elif selected_tool == ToolType.WEB_SEARCH.value:
                metrics = await self._evaluate_web(state)
            else:
                logger.warning(f"Unknown tool type: {selected_tool}, skipping evaluation")
                return

            await self._save_to_database(
                session_id=session_id,
                user_query=user_query,
                current_response=current_response,
                tool_type=selected_tool,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Background evaluation failed: {e}", exc_info=True)

    async def _evaluate_rag(self, state: WorkflowState) -> Dict[str, Any]:
        current_response = state.get("current_synthesized_response", "")
        rag_documents = state.get("retrieved_documents_with_scores", [])
        rag_context = state.get("rag_synthesizer_response", "")
        user_query = state.get("user_query", "")

        return await self.rag_evaluator.evaluate(
            current_response=current_response,
            rag_documents=rag_documents,
            rag_context=rag_context,
            user_query=user_query
        )

    async def _evaluate_web(self, state: WorkflowState) -> Dict[str, Any]:
        current_response = state.get("current_synthesized_response", "")
        web_results = state.get("web_search_results", [])
        user_query = state.get("user_query", "")

        return await self.web_evaluator.evaluate(
            current_response=current_response,
            web_results=web_results,
            user_query=user_query
        )

    async def _save_to_database(
        self,
        session_id: str,
        user_query: str,
        current_response: str,
        tool_type: str,
        metrics: Dict[str, Any]
    ) -> None:
        try:
            await self.evaluation_repo.save_evaluation(
                session_id=session_id,
                user_query=user_query,
                tool_type=tool_type,
                confidence_score=metrics["confidence_score"],
                current_response=current_response,
                faithfulness=metrics.get("faithfulness"),
                retrieval_quality=metrics.get("retrieval_quality"),
                factual_consistency=metrics.get("factual_consistency"),
                relevance_score=metrics.get("relevance_score"),
                additional_metadata=metrics.get("metadata", {})
            )
            logger.info(f"Successfully saved evaluation for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save evaluation to database: {e}")
            raise


# Singleton instance
_evaluation_service: Optional[EvaluationService] = None


def get_evaluation_service() -> EvaluationService:
    """Get or create the singleton EvaluationService instance."""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service