import numpy as np
from typing import List
from langchain_core.documents import Document
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes


class ConfidenceEvaluatorAgent:

    def __init__(
        self,
        min_confidence_threshold: float = 0.7,
        min_documents_threshold: int = 2,
        avg_score_threshold: float = 0.5,
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_documents_threshold = min_documents_threshold
        self.avg_score_threshold = avg_score_threshold

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            documents: List[Document] = state.get("retrieved_documents", [])
            scores: List[float] = state.get("retrieval_scores", [])

            confidence_score = self._calculate_confidence(documents, scores)
            needs_web_search = confidence_score < self.min_confidence_threshold

            logger.info(
                f"[ConfidenceEvaluatorAgent] Confidence score: {confidence_score:.3f} "
                f"(threshold: {self.min_confidence_threshold})"
            )
            logger.info(
                f"[ConfidenceEvaluatorAgent] Decision: "
                f"{'USE WEB SEARCH' if needs_web_search else 'RAG SUFFICIENT'}"
            )

            return {
                **state,
                "confidence_score": confidence_score,
                "needs_web_search": needs_web_search,
            }

        except Exception as e:
            logger.error(
                f"[ConfidenceEvaluatorAgent] Error during evaluation: {e}",
                exc_info=True
            )
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"ConfidenceEvaluatorAgent error: [{type(e).__name__}]: {str(e)}",
            )

    def _calculate_confidence(self, documents: List[Document], scores: List[float]) -> float:
        
        num_docs = len(documents)
        if num_docs == 0:
            return 0.0

        doc_count_score = min(num_docs / self.min_documents_threshold, 1.0)

        if scores and len(scores) > 0:
            avg_score = np.mean([s for s in scores if s > 0])
            score_confidence = min(avg_score / self.avg_score_threshold, 1.0)
        else:
            score_confidence = 0.6

        content_lengths = [len(doc.page_content) for doc in documents]
        avg_length = np.mean(content_lengths) if content_lengths else 0
        content_score = min(avg_length / 100, 1.0)

        unique_sources = len(set(
            doc.metadata.get("source", "unknown") for doc in documents
        ))
        diversity_score = min(unique_sources / max(num_docs * 0.5, 1), 1.0)

        confidence = (
            0.3 * doc_count_score +
            0.4 * score_confidence +
            0.2 * content_score +
            0.1 * diversity_score
        )

        logger.debug(
            f"[ConfidenceEvaluatorAgent] Score breakdown: "
            f"docs={doc_count_score:.2f}, "
            f"relevance={score_confidence:.2f}, "
            f"content={content_score:.2f}, "
            f"diversity={diversity_score:.2f}"
        )

        return round(confidence, 3)
