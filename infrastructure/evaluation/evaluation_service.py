import asyncio
import logging
from typing import Dict, Any
from domain.enums.workflow_state import WorkflowState, ToolType
from agent.rag_evaluator import RAGEvaluator
from agent.web_evaluator import WebEvaluator
from infrastructure.database.connection import db_connection
from infrastructure.database.models import EvaluationMetrics

logger = logging.getLogger(__name__)

class EvaluationService:
    
    def __init__(self):
        self.rag_evaluator = RAGEvaluator()
        self.web_evaluator = WebEvaluator()

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
            async with db_connection.async_session_factory() as session:
                evaluation = EvaluationMetrics(
                    session_id=session_id,
                    user_query=user_query,
                    tool_type=tool_type,
                    faithfulness=metrics.get("faithfulness"),
                    retrieval_quality=metrics.get("retrieval_quality"),
                    factual_consistency=metrics.get("factual_consistency"),
                    relevance_score=metrics.get("relevance_score"),
                    confidence_score=metrics["confidence_score"],
                    current_response=current_response,
                    additional_metadata=metrics.get("metadata", {})
                )

                session.add(evaluation)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to save evaluation to database: {e}")
            raise

evaluation_service = EvaluationService()