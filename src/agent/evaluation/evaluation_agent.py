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
            current_response = state.get("current_synthesized_response", "")

            if not current_response:
                return {
                    **state,
                    "evaluation_metrics": {"skipped": True, "reason": "No response to evaluate"}
                }

            if selected_tool == ToolType.RAG_SEARCH.value:
                metrics = await self._evaluate_rag_response(state)
            elif selected_tool == ToolType.WEB_SEARCH.value:
                metrics = await self._evaluate_web_response(state)
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
        current_response = state.get("current_synthesized_response", "")
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
        current_response = state.get("current_synthesized_response", "")
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

    async def _check_faithfulness(self, answer: str, rag_context: str, rag_documents: List) -> str:
        # Returns: supported | partial | unsupported
        try:
            formatted_context = self._format_rag_documents(rag_documents)
            if not formatted_context and rag_context:
                formatted_context = rag_context

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
        try:
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

        faith_score = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "error": 0.3, "skipped": 0.5}.get(faithfulness, 0.5)
        rel_score = {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0, "error": 0.3, "skipped": 0.5}.get(answer_relevance, 0.5)
        prec_score = {"precise": 1.0, "moderate": 0.6, "unfocused": 0.2, "error": 0.3, "skipped": 0.5}.get(context_precision, 0.5)

        return float(0.35 * faith_score + 0.25 * rel_score + 0.20 * prec_score + 0.20 * retrieval_quality)

    def _calculate_full_confidence_web(self, metrics: Dict[str, Any]) -> float:
        consistency = metrics.get("factual_consistency", "partial")
        answer_relevance = metrics.get("answer_relevance", "partial")
        relevance_score = metrics.get("relevance_score", 0.5)

        cons_score = {"consistent": 1.0, "partial": 0.5, "unsupported": 0.0, "error": 0.3, "skipped": 0.5}.get(consistency, 0.5)
        rel_score = {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0, "error": 0.3, "skipped": 0.5}.get(answer_relevance, 0.5)

        return float(0.40 * cons_score + 0.30 * rel_score + 0.30 * relevance_score)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1, vec2 = np.array(vec1), np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

    def _format_rag_documents(self, rag_documents: List) -> str:
        if not rag_documents:
            return "No RAG documents available."
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
            return "No web search results available."
        context_parts = []
        for i, doc in enumerate(web_results, 1):
            title = doc.metadata.get("title", "Untitled")
            url = doc.metadata.get("url", "")
            context_parts.append(f"[Web Result {i}]\nTitle: {title}\nURL: {url}\nContent: {doc.page_content}\n")
        return "\n".join(context_parts)
