from typing import List, Dict, Any
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from prompts.evaluation_prompts import RAG_FAITHFULNESS_PROMPT
import logging

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        self.llm = loadLLM(LLMType.ORCHESTRATION_SYNTHESIZER_AGENT)
        self.faithfulness_chain = RAG_FAITHFULNESS_PROMPT | self.llm | StrOutputParser()

    async def evaluate(
        self,
        current_response: str,
        rag_documents: List,
        rag_context: str,
        user_query: str
    ) -> Dict[str, Any]:
        try:
            faithfulness = await self._check_faithfulness(
                current_response,
                rag_context,
                rag_documents
            )
    
            retrieval_quality = self._calculate_retrieval_quality(rag_documents)
            confidence_score = self._calculate_confidence_score(
                faithfulness=faithfulness,
                retrieval_quality=retrieval_quality
            )

            return {
                "faithfulness": faithfulness,
                "retrieval_quality": retrieval_quality,
                "confidence_score": confidence_score,
                "metadata": {
                    "num_documents": len(rag_documents),
                    "tool_type": "rag_search"
                }
            }

        except Exception as e:
            logger.error(f"RAG evaluation error: {e}")
            return {
                "faithfulness": "error",
                "retrieval_quality": 0.0,
                "confidence_score": 0.0,
                "metadata": {"error": str(e)}
            }

    async def _check_faithfulness(
        self,
        answer: str,
        rag_context: str,
        rag_documents: List
    ) -> str:
        try:
            formatted_context = self._format_rag_documents(rag_documents)
            if not formatted_context and rag_context:
                formatted_context = rag_context

            result = await self.faithfulness_chain.ainvoke({
                "rag_context": formatted_context,
                "answer": answer
            })

            result = result.strip().lower()
            if "supported" in result and "unsupported" not in result:
                return "supported"
            elif "partial" in result:
                return "partial"
            elif "unsupported" in result:
                return "unsupported"
            else:
                return "partial"

        except Exception as e:
            logger.error(f"Faithfulness check error: {e}")
            return "error"

    def _calculate_retrieval_quality(self, rag_documents: List) -> float:
        if not rag_documents:
            return 0.0

        scores = []
        for doc_item in rag_documents:
            if isinstance(doc_item, dict) and "score" in doc_item:
                scores.append(doc_item["score"])

        if not scores:
            return 0.5

        return float(np.mean(scores))

    def _calculate_confidence_score(
        self,
        faithfulness: str,
        retrieval_quality: float
    ) -> float:
        faithfulness_score = {
            "supported": 1.0,
            "partial": 0.5,
            "unsupported": 0.0,
            "error": 0.3
        }.get(faithfulness, 0.5)

        confidence = (0.6 * faithfulness_score) + (0.4 * retrieval_quality)
        return float(confidence)

    def _format_rag_documents(self, rag_documents: List) -> str:
        """Format RAG documents for evaluation."""
        if not rag_documents:
            return "No RAG documents available."

        context_parts = []
        for i, doc_item in enumerate(rag_documents, 1):
            if isinstance(doc_item, dict) and "document" in doc_item:
                doc = doc_item["document"]
                content = doc.page_content
                context_parts.append(f"[Document {i}]\n{content}\n")
            else:
                content = doc_item.page_content if hasattr(doc_item, 'page_content') else str(doc_item)
                context_parts.append(f"[Document {i}]\n{content}\n")

        return "\n".join(context_parts)