from typing import List, Dict, Any
import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from src.graph.state import WorkflowState, ToolType
from src.constants import LLMAgentName, LLMType
from src.infras import llm_loader
from src.config import config
from src.agent.base import AgentInterface
from src.prompts.evaluation import (
    RAG_FAITHFULNESS_PROMPT,
    WEB_CONSISTENCY_PROMPT,
    ANSWER_RELEVANCE_PROMPT,
    CONTEXT_PRECISION_PROMPT,
)
from src.constants.evaluation import CONSISTENCY_SCORES, EMPTY_RAG_CONTEXT, EMPTY_WEB_CONTEXT, FAITHFULNESS_SCORES, PRECISION_SCORES, RELEVANCE_SCORES

class EvaluationAgent(AgentInterface):
    def __init__(self):
        self._agent_name = LLMAgentName.EVALUATION_AGENT.value

        # Initialize embeddings for quantitative metrics
        embedding_config = config.get("embedding", {})
        openai_config = embedding_config.get("openai", {})
        embedding_model = openai_config.get("model_name", "text-embedding-3-small")
        embedder_api_key = openai_config.get("api_key")
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=embedder_api_key
        )

        # Initialize LLM chains for all evaluation checks
        self.llm = llm_loader.loadLLM(LLMType.EVALUATION_AGENT)
        self.faithfulness_chain = RAG_FAITHFULNESS_PROMPT | self.llm | StrOutputParser()
        self.consistency_chain = WEB_CONSISTENCY_PROMPT | self.llm | StrOutputParser()
        self.relevance_chain = ANSWER_RELEVANCE_PROMPT | self.llm | StrOutputParser()
        self.precision_chain = CONTEXT_PRECISION_PROMPT | self.llm | StrOutputParser()

    @property
    def name(self) -> str:
        return self._agent_name or super().name

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            selected_tool = state.get("selected_tool", "")
            current_response = state.get("response", "")

            if not current_response:
                return {
                    **state,
                    "evaluation_metrics": {"skipped": True, "reason": "No response to evaluate"}
                }

            if selected_tool == ToolType.RAG_SEARCH.value:
                metrics = await self._evaluate_rag_response(state)
            elif selected_tool == ToolType.WEB_SEARCH.value:
                metrics = await self._evaluate_web_response(state)
            elif selected_tool == ToolType.HYBRID_SEARCH.value:
                metrics = await self._evaluate_hybrid_response(state)
            else:
                metrics = {"skipped": True, "reason": f"Unknown tool: {selected_tool}"}

            return {
                **state,
                "evaluation_metrics": metrics,
                "faithfulness": metrics.get("faithfulness", ""),
                "factual_consistency": metrics.get("factual_consistency", ""),
                "answer_relevance": metrics.get("answer_relevance", ""),
                "context_precision": metrics.get("context_precision", ""),
                "retrieval_quality": metrics.get("retrieval_quality", 0.0),
                "relevance_score": metrics.get("relevance_score", 0.0),
                "confidence_score": metrics.get("confidence_score", 0.0),
            }

        except Exception as e:
            return {
                **state,
                "evaluation_metrics": {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            }

    async def _evaluate_rag_response(self, state: WorkflowState) -> Dict[str, Any]:
        current_response = state.get("response", "")
        rag_documents = state.get("retrieved_documents_with_scores", [])
        rag_context = state.get("rag_synthesizer_response", "")
        user_query = state.get("user_query", "")

        metrics = {"tool_type": "rag_search"}

        # Quantitative metrics
        retrieval_quality = self._calculate_retrieval_quality(rag_documents)
        metrics["retrieval_quality"] = retrieval_quality

        # LLM-based metrics (always enabled)
        metrics["faithfulness"] = await self._check_faithfulness(current_response, rag_context, rag_documents)
        metrics["answer_relevance"] = await self._check_answer_relevance(user_query, current_response)
        metrics["context_precision"] = await self._check_context_precision(user_query, rag_context, rag_documents)

        # Calculate comprehensive confidence score
        metrics["confidence_score"] = self._calculate_full_confidence_rag(metrics)
        metrics["num_documents"] = len(rag_documents)
        
        return metrics

    async def _evaluate_web_response(self, state: WorkflowState) -> Dict[str, Any]:
        current_response = state.get("response", "")
        web_results = state.get("web_search_results", [])
        user_query = state.get("user_query", "")
        web_context = self._format_web_context(web_results)

        metrics = {"tool_type": "web_search"}

        # Quantitative metrics
        relevance_score = await self._calculate_relevance(user_query, web_results)
        metrics["relevance_score"] = relevance_score

        # LLM-based metrics (always enabled)
        metrics["factual_consistency"] = await self._check_consistency(current_response, web_context)
        metrics["answer_relevance"] = await self._check_answer_relevance(user_query, current_response)

        # Calculate comprehensive confidence score
        metrics["confidence_score"] = self._calculate_full_confidence_web(metrics)
        metrics["num_web_results"] = len(web_results)
        
        return metrics

    async def _evaluate_hybrid_response(self, state: WorkflowState) -> Dict[str, Any]:
        current_response = state.get("response", "")
        rag_documents = state.get("retrieved_documents_with_scores", [])
        web_results = state.get("web_search_results", [])
        user_query = state.get("user_query", "")
        
        # Format contexts for evaluation
        rag_context = state.get("rag_synthesizer_response", "")
        if not rag_context:
            rag_context = self._format_rag_documents(rag_documents)
        web_context = self._format_web_context(web_results)

        # Track which sources contributed
        has_rag = bool(rag_documents and rag_context and rag_context != EMPTY_RAG_CONTEXT)
        has_web = bool(web_results and web_context and web_context != EMPTY_WEB_CONTEXT)

        metrics = {
            "tool_type": "hybrid_search",
            "has_rag_contribution": has_rag,
            "has_web_contribution": has_web
        }

        # Quantitative metrics for both sources
        retrieval_quality = self._calculate_retrieval_quality(rag_documents)
        relevance_score = await self._calculate_relevance(user_query, web_results)
        metrics["retrieval_quality"] = retrieval_quality
        metrics["relevance_score"] = relevance_score

        # RAG-specific metrics (only if RAG data available)
        if has_rag:
            metrics["faithfulness"] = await self._check_faithfulness(current_response, rag_context, rag_documents)
            metrics["context_precision"] = await self._check_context_precision(user_query, rag_context, rag_documents)
        else:
            metrics["faithfulness"] = "skipped"
            metrics["context_precision"] = "skipped"

        # Web-specific metrics (only if web data available)
        if has_web:
            metrics["factual_consistency"] = await self._check_consistency(current_response, web_context)
        else:
            metrics["factual_consistency"] = "skipped"

        # Overall answer relevance (evaluates entire response)
        metrics["answer_relevance"] = await self._check_answer_relevance(user_query, current_response)

        # Calculate comprehensive hybrid confidence score
        metrics["confidence_score"] = self._calculate_full_confidence_hybrid(metrics, has_rag, has_web)
        
        metrics["num_documents"] = len(rag_documents)
        metrics["num_web_results"] = len(web_results)
        
        return metrics

    async def _check_faithfulness(self, answer: str, rag_context: str, rag_documents: List) -> str:
        """Evaluate if the answer is supported by RAG context."""
        try:
            formatted_context = self._format_rag_documents(rag_documents)
            if not formatted_context and rag_context:
                formatted_context = rag_context

            # Skip if no context available
            if not formatted_context or formatted_context == EMPTY_RAG_CONTEXT:
                return "unsupported"

            result = await self.faithfulness_chain.ainvoke({"rag_context": formatted_context, "answer": answer})
            result = result.strip().lower()
            if "supported" in result and "unsupported" not in result:
                return "supported"
            elif "partial" in result:
                return "partial"
            elif "unsupported" in result:
                return "unsupported"
            return "partial"
        except Exception:
            return "error"

    async def _check_consistency(self, answer: str, web_context: str) -> str:
        """Evaluate factual consistency between answer and web context."""
        try:
            # Skip if no context available
            if not web_context or web_context == EMPTY_WEB_CONTEXT:
                return "unsupported"

            result = await self.consistency_chain.ainvoke({"web_context": web_context, "answer": answer})
            result = result.strip().lower()
            if "consistent" in result and "inconsistent" not in result:
                return "consistent"
            elif "partial" in result:
                return "partial"
            elif "unsupported" in result or "inconsistent" in result:
                return "unsupported"
            return "partial"
        except Exception:
            return "error"

    async def _check_answer_relevance(self, query: str, answer: str) -> str:
        try:
            result = await self.relevance_chain.ainvoke({"query": query, "answer": answer})
            result = result.strip().lower()
            if "relevant" in result and "irrelevant" not in result:
                return "relevant"
            elif "partial" in result:
                return "partial"
            elif "irrelevant" in result:
                return "irrelevant"
            return "partial"
        except Exception:
            return "error"

    async def _check_context_precision(self, query: str, rag_context: str, rag_documents: List) -> str:
        
        try:
            formatted_context = self._format_rag_documents(rag_documents)
            if not formatted_context and rag_context:
                formatted_context = rag_context

            # Skip if no context available
            if not formatted_context or formatted_context == EMPTY_RAG_CONTEXT:
                return "unfocused"

            result = await self.precision_chain.ainvoke({"query": query, "rag_context": formatted_context})
            result = result.strip().lower()
            if "precise" in result:
                return "precise"
            elif "moderate" in result:
                return "moderate"
            elif "unfocused" in result:
                return "unfocused"
            return "moderate"
        except Exception:
            return "error"

    def _calculate_retrieval_quality(self, rag_documents: List) -> float:
        if not rag_documents:
            return 0.0
        scores = []
        for doc_item in rag_documents:
            if isinstance(doc_item, dict) and "score" in doc_item:
                scores.append(doc_item["score"])
        return float(np.mean(scores)) if scores else 0.5

    async def _calculate_relevance(self, query: str, web_results: List[Document]) -> float:
        # Embedding similarity between query and web results
        if not web_results or not query:
            return 0.0
        try:
            query_embedding = await self.embeddings.aembed_query(query)
            web_texts = [doc.page_content for doc in web_results[:5]]
            if not web_texts:
                return 0.0
            web_embeddings = await self.embeddings.aembed_documents(web_texts)
            similarities = [self._cosine_similarity(query_embedding, web_emb) for web_emb in web_embeddings]
            return float(np.mean(similarities)) if similarities else 0.0
        except Exception:
            return 0.5

    def _calculate_full_confidence_rag(self, metrics: Dict[str, Any]) -> float:
        faithfulness = metrics.get("faithfulness", "partial")
        answer_relevance = metrics.get("answer_relevance", "partial")
        context_precision = metrics.get("context_precision", "moderate")
        retrieval_quality = metrics.get("retrieval_quality", 0.5)

        faith_score = FAITHFULNESS_SCORES.get(faithfulness, 0.5)
        rel_score = RELEVANCE_SCORES.get(answer_relevance, 0.5)
        prec_score = PRECISION_SCORES.get(context_precision, 0.5)

        return float(0.35 * faith_score + 0.25 * rel_score + 0.20 * prec_score + 0.20 * retrieval_quality)

    def _calculate_full_confidence_web(self, metrics: Dict[str, Any]) -> float:
        """Calculate confidence score for web-only search."""
        consistency = metrics.get("factual_consistency", "partial")
        answer_relevance = metrics.get("answer_relevance", "partial")
        relevance_score = metrics.get("relevance_score", 0.5)

        cons_score = CONSISTENCY_SCORES.get(consistency, 0.5)
        rel_score = RELEVANCE_SCORES.get(answer_relevance, 0.5)

        return float(0.40 * cons_score + 0.30 * rel_score + 0.30 * relevance_score)

    def _calculate_full_confidence_hybrid(self, metrics: Dict[str, Any], has_rag: bool, has_web: bool) -> float:
        """Calculate confidence score for hybrid search with dynamic weighting based on source availability."""
        faithfulness = metrics.get("faithfulness", "partial")
        context_precision = metrics.get("context_precision", "moderate")
        factual_consistency = metrics.get("factual_consistency", "partial")
        answer_relevance = metrics.get("answer_relevance", "partial")
        retrieval_quality = metrics.get("retrieval_quality", 0.5)
        relevance_score = metrics.get("relevance_score", 0.5)

        faith_score = FAITHFULNESS_SCORES.get(faithfulness, 0.5)
        prec_score = PRECISION_SCORES.get(context_precision, 0.5)
        cons_score = CONSISTENCY_SCORES.get(factual_consistency, 0.5)
        rel_score = RELEVANCE_SCORES.get(answer_relevance, 0.5)

        # Adjust weights dynamically based on which sources contributed
        if has_rag and has_web:
            # Both sources: balanced weighting
            return float(
                0.20 * faith_score +
                0.15 * prec_score +
                0.15 * retrieval_quality +
                0.20 * cons_score +
                0.15 * relevance_score +
                0.15 * rel_score
            )
        elif has_rag:
            # Only RAG: weight RAG metrics more heavily
            return float(
                0.40 * faith_score +
                0.25 * prec_score +
                0.20 * retrieval_quality +
                0.15 * rel_score
            )
        elif has_web:
            # Only Web: weight web metrics more heavily
            return float(
                0.45 * cons_score +
                0.30 * relevance_score +
                0.25 * rel_score
            )
        else:
            # No sources (shouldn't happen in hybrid, but handle gracefully)
            return 0.0

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1, vec2 = np.array(vec1), np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

    def _format_rag_documents(self, rag_documents: List) -> str:
        if not rag_documents:
            return EMPTY_RAG_CONTEXT
        context_parts = []
        for i, doc_item in enumerate(rag_documents, 1):
            if isinstance(doc_item, dict) and "document" in doc_item:
                content = doc_item["document"].page_content
            else:
                content = doc_item.page_content if hasattr(doc_item, 'page_content') else str(doc_item)
            context_parts.append(f"[Document {i}]\n{content}\n")
        return "\n".join(context_parts)

    def _format_web_context(self, web_results: List[Document]) -> str:
        if not web_results:
            return EMPTY_WEB_CONTEXT
        context_parts = []
        for i, doc in enumerate(web_results, 1):
            title = doc.metadata.get("title", "Untitled")
            url = doc.metadata.get("url", "")
            context_parts.append(f"[Web Result {i}]\nTitle: {title}\nURL: {url}\nContent: {doc.page_content}\n")
        return "\n".join(context_parts)
