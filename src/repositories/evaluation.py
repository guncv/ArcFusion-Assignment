from typing import Optional, Dict, Any
from src.models.evaluate_metric import EvaluationMetrics
from src.infras.database.postgres import postgres_database
from src.infras.log import logger

class EvaluationRepository:
    def __init__(self):
        self.db = postgres_database

    async def save_evaluation(
        self,
        session_id: str,
        user_query: str,
        tool_type: str,
        confidence_score: float,
        current_response: Optional[str] = None,
        faithfulness: Optional[str] = None,
        retrieval_quality: Optional[float] = None,
        factual_consistency: Optional[str] = None,
        relevance_score: Optional[float] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[EvaluationMetrics]:
        try:
            async with self.db.async_session_factory() as session:
                evaluation = EvaluationMetrics(
                    session_id=session_id,
                    user_query=user_query,
                    tool_type=tool_type,
                    confidence_score=confidence_score,
                    current_response=current_response,
                    faithfulness=faithfulness,
                    retrieval_quality=retrieval_quality,
                    factual_consistency=factual_consistency,
                    relevance_score=relevance_score,
                    additional_metadata=additional_metadata
                )

                session.add(evaluation)
                await session.commit()
                await session.refresh(evaluation)

                logger.debug(f"Saved evaluation for session {session_id}, tool: {tool_type}")
                return evaluation

        except Exception as e:
            logger.error(f"Failed to save evaluation metrics: {e}")
            return None

_evaluation_repo: Optional[EvaluationRepository] = None

def get_evaluation_repository() -> EvaluationRepository:
    global _evaluation_repo
    if _evaluation_repo is None:
        _evaluation_repo = EvaluationRepository()
    return _evaluation_repo
