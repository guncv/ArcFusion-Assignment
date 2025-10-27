from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.state import WorkflowState

from src.agent.evaluation import EvaluationAgent
from src.repositories.evaluation import get_evaluation_repository
from src.infras.log import logger

class EvaluationService:

    def __init__(self):
        self.evaluator = EvaluationAgent()
        self.evaluation_repo = get_evaluation_repository()

    async def evaluate_and_save(self, state: 'WorkflowState') -> None:
        try:
            selected_tool = state.get("selected_tool", "")
            current_response = state.get("current_synthesized_response", "")
            session_id = state.get("session_id", "unknown")
            user_query = state.get("user_query", "")

            if not current_response:
                logger.warning("No current_response to evaluate, skipping")
                return

            metrics = await self.evaluator.ainvoke(state)

            await self._save_to_database(
                session_id=session_id,
                user_query=user_query,
                current_response=current_response,
                tool_type=selected_tool,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Background evaluation failed: {e}", exc_info=True)
            
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

_evaluation_service: Optional[EvaluationService] = None

def get_evaluation_service() -> EvaluationService:
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service