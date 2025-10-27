from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import select, delete, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
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

    async def get_evaluations_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[EvaluationMetrics]:
        try:
            async with self.db.async_session_factory() as session:
                query = (
                    select(EvaluationMetrics)
                    .where(EvaluationMetrics.session_id == session_id)
                    .order_by(EvaluationMetrics.created_at.desc())
                )

                if limit:
                    query = query.limit(limit)

                result = await session.execute(query)
                evaluations = result.scalars().all()

                return list(evaluations)

        except Exception as e:
            logger.error(f"Failed to retrieve evaluations: {e}")
            return []

    async def get_evaluation_by_id(
        self,
        evaluation_id: str
    ) -> Optional[EvaluationMetrics]:
        try:
            async with self.db.async_session_factory() as session:
                result = await session.execute(
                    select(EvaluationMetrics)
                    .where(EvaluationMetrics.id == evaluation_id)
                )
                evaluation = result.scalar_one_or_none()

                return evaluation

        except Exception as e:
            logger.error(f"Failed to retrieve evaluation: {e}")
            return None

    async def get_evaluations_by_tool(
        self,
        tool_type: str,
        limit: Optional[int] = None
    ) -> List[EvaluationMetrics]:
        try:
            async with self.db.async_session_factory() as session:
                query = (
                    select(EvaluationMetrics)
                    .where(EvaluationMetrics.tool_type == tool_type)
                    .order_by(EvaluationMetrics.created_at.desc())
                )

                if limit:
                    query = query.limit(limit)

                result = await session.execute(query)
                evaluations = result.scalars().all()

                return list(evaluations)

        except Exception as e:
            logger.error(f"Failed to retrieve evaluations by tool: {e}")
            return []

    async def get_average_confidence(
        self,
        session_id: Optional[str] = None,
        tool_type: Optional[str] = None,
        hours: Optional[int] = None
    ) -> float:
        try:
            async with self.db.async_session_factory() as session:
                query = select(func.avg(EvaluationMetrics.confidence_score))

                conditions = []
                if session_id:
                    conditions.append(EvaluationMetrics.session_id == session_id)
                if tool_type:
                    conditions.append(EvaluationMetrics.tool_type == tool_type)
                if hours:
                    time_threshold = datetime.utcnow() - timedelta(hours=hours)
                    conditions.append(EvaluationMetrics.created_at >= time_threshold)

                if conditions:
                    query = query.where(and_(*conditions))

                result = await session.execute(query)
                avg_confidence = result.scalar()

                return float(avg_confidence) if avg_confidence else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate average confidence: {e}")
            return 0.0

    async def get_statistics(
        self,
        tool_type: Optional[str] = None,
        hours: Optional[int] = 24
    ) -> Dict[str, Any]:
        try:
            async with self.db.async_session_factory() as session:
                time_threshold = datetime.utcnow() - timedelta(hours=hours)

                query = select(
                    func.count(EvaluationMetrics.id).label('total'),
                    func.avg(EvaluationMetrics.confidence_score).label('avg_confidence'),
                    func.min(EvaluationMetrics.confidence_score).label('min_confidence'),
                    func.max(EvaluationMetrics.confidence_score).label('max_confidence')
                ).where(EvaluationMetrics.created_at >= time_threshold)

                if tool_type:
                    query = query.where(EvaluationMetrics.tool_type == tool_type)

                result = await session.execute(query)
                stats = result.one()

                return {
                    'total_evaluations': stats.total or 0,
                    'average_confidence': float(stats.avg_confidence) if stats.avg_confidence else 0.0,
                    'min_confidence': float(stats.min_confidence) if stats.min_confidence else 0.0,
                    'max_confidence': float(stats.max_confidence) if stats.max_confidence else 0.0,
                    'time_window_hours': hours
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'total_evaluations': 0,
                'average_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'time_window_hours': hours
            }

    async def delete_evaluation(self, evaluation_id: str) -> bool:
        try:
            async with self.db.async_session_factory() as session:
                await session.execute(
                    delete(EvaluationMetrics).where(EvaluationMetrics.id == evaluation_id)
                )
                await session.commit()

                logger.debug(f"Deleted evaluation {evaluation_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete evaluation: {e}")
            return False

    async def delete_old_evaluations(self, days: int = 30) -> int:
        try:
            async with self.db.async_session_factory() as session:
                time_threshold = datetime.utcnow() - timedelta(days=days)

                result = await session.execute(
                    delete(EvaluationMetrics)
                    .where(EvaluationMetrics.created_at < time_threshold)
                    .returning(EvaluationMetrics.id)
                )
                deleted_count = len(result.all())
                await session.commit()

                logger.info(f"Deleted {deleted_count} old evaluations (older than {days} days)")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete old evaluations: {e}")
            return 0


_evaluation_repo: Optional[EvaluationRepository] = None

def get_evaluation_repository() -> EvaluationRepository:
    global _evaluation_repo
    if _evaluation_repo is None:
        _evaluation_repo = EvaluationRepository()
    return _evaluation_repo
