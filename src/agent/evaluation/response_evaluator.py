from typing import List, Dict, Any
import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from src.graph.state import WorkflowState, ToolType
from src.constants.llm_type import LLMType
from src.infras import llm_loader
from src.config import config
from src.prompts import (
    RAG_FAITHFULNESS_PROMPT,
    WEB_CONSISTENCY_PROMPT
)
from src.utils.exception import ArcFusionException
from src.constants.error_code import ArcFusionErrorCodes

class ResponseEvaluator:
    """
    Evaluates response quality for monitoring in production.
    Runs asynchronously after synthesis to provide quality metrics.
    """

    def __init__(self):
        self.llm = llm_loader.loadLLM(LLMType.SYNTHESIZER_AGENT)  # Use same LLM for evaluation

        # Initialize embeddings for relevance calculations
        embedding_config = config.get("embedding", {})
        openai_config = embedding_config.get("openai", {})
        embedding_model = openai_config.get("model_name", "text-embedding-3-small")
        embedder_api_key = openai_config.get("api_key")
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=embedder_api_key
        )

        # Evaluation chains
        self.faithfulness_chain = RAG_FAITHFULNESS_PROMPT | self.llm | StrOutputParser()
        self.consistency_chain = WEB_CONSISTENCY_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate the current synthesized response based on the tool used.
        """
        try:
            selected_tool = state.get("selected_tool", "")
            current_response = state.get("current_synthesized_response", "")

            # Skip evaluation if no response
            if not current_response:
                return {
                    **state,
                    "evaluation_metrics": {"skipped": True, "reason": "No response to evaluate"}
                }

            # Evaluate based on tool type
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
                "retrieval_quality": metrics.get("retrieval_quality", 0.0),
                "relevance_score": metrics.get("relevance_score", 0.0),
                "confidence_score": metrics.get("confidence_score", 0.0),
            }

        except Exception as e:
            # Don't fail the whole workflow if evaluation fails
            # Just log the error and continue
            return {
                **state,
                "evaluation_metrics": {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            }

    async def _evaluate_rag_response(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Evaluate RAG response:
        1. Faithfulness/Groundedness (LLM judge)
        2. Retrieval Quality (cosine similarity)
        3. Confidence Score (combined)
        """
        current_response = state.get("current_synthesized_response", "")
        rag_documents = state.get("retrieved_documents_with_scores", [])
        rag_context = state.get("rag_synthesizer_response", "")
        user_query = state.get("user_query", "")

        # 1. Faithfulness Check (LLM Judge)
        faithfulness = await self._check_faithfulness(current_response, rag_context, rag_documents)

        # 2. Retrieval Quality (average similarity from retrieved docs)
        retrieval_quality = self._calculate_retrieval_quality(rag_documents)

        # 3. Confidence Score (combined metric)
        confidence_score = self._calculate_confidence_score(
            faithfulness=faithfulness,
            retrieval_quality=retrieval_quality,
            tool_type="rag"
        )

        return {
            "tool_type": "rag_search",
            "faithfulness": faithfulness,
            "retrieval_quality": retrieval_quality,
            "confidence_score": confidence_score,
            "num_documents": len(rag_documents)
        }

    async def _evaluate_web_response(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Evaluate WebSearch response:
        1. Factual Consistency (LLM judge)
        2. Relevance/Coverage (embedding similarity)
        3. Confidence Score (combined)
        """
        current_response = state.get("current_synthesized_response", "")
        web_results = state.get("web_search_results", [])
        user_query = state.get("user_query", "")

        # Format web context for evaluation
        web_context = self._format_web_context(web_results)

        # 1. Factual Consistency Check (LLM Judge)
        consistency = await self._check_consistency(current_response, web_context)

        # 2. Relevance/Coverage (embedding similarity)
        relevance_score = await self._calculate_relevance(user_query, web_results)

        # 3. Confidence Score (combined metric)
        confidence_score = self._calculate_confidence_score(
            consistency=consistency,
            relevance=relevance_score,
            tool_type="web"
        )

        return {
            "tool_type": "web_search",
            "factual_consistency": consistency,
            "relevance_score": relevance_score,
            "confidence_score": confidence_score,
            "num_web_results": len(web_results)
        }

    async def _check_faithfulness(self, answer: str, rag_context: str, rag_documents: List) -> str:
        """
        Use LLM judge to check if answer is grounded in RAG context.
        Returns: "supported" | "partial" | "unsupported"
        """
        try:
            # Format RAG context
            formatted_context = self._format_rag_documents(rag_documents)
            if not formatted_context and rag_context:
                formatted_context = rag_context

            result = await self.faithfulness_chain.ainvoke({
                "rag_context": formatted_context,
                "answer": answer
            })

            # Clean and normalize result
            result = result.strip().lower()
            if "supported" in result and "unsupported" not in result:
                return "supported"
            elif "partial" in result:
                return "partial"
            elif "unsupported" in result:
                return "unsupported"
            else:
                return "partial"  # Default to partial if unclear

        except Exception as e:
            return "error"

    async def _check_consistency(self, answer: str, web_context: str) -> str:
        """
        Use LLM judge to check if answer is consistent with web results.
        Returns: "consistent" | "partial" | "unsupported"
        """
        try:
            result = await self.consistency_chain.ainvoke({
                "web_context": web_context,
                "answer": answer
            })

            # Clean and normalize result
            result = result.strip().lower()
            if "consistent" in result and "inconsistent" not in result:
                return "consistent"
            elif "partial" in result:
                return "partial"
            elif "unsupported" in result or "inconsistent" in result:
                return "unsupported"
            else:
                return "partial"  # Default to partial if unclear

        except Exception as e:
            return "error"

    def _calculate_retrieval_quality(self, rag_documents: List) -> float:
        """
        Calculate average retrieval quality from document scores.
        Returns: 0-1 score
        """
        if not rag_documents:
            return 0.0

        scores = []
        for doc_item in rag_documents:
            if isinstance(doc_item, dict) and "score" in doc_item:
                scores.append(doc_item["score"])

        if not scores:
            return 0.5  # Default if no scores available

        return float(np.mean(scores))

    async def _calculate_relevance(self, query: str, web_results: List[Document]) -> float:
        """
        Calculate relevance using embedding similarity between query and web results.
        Returns: 0-1 score
        """
        if not web_results or not query:
            return 0.0

        try:
            # Get query embedding
            query_embedding = await self.embeddings.aembed_query(query)

            # Get embeddings for web results
            web_texts = [doc.page_content for doc in web_results[:5]]  # Top 5 results
            if not web_texts:
                return 0.0

            web_embeddings = await self.embeddings.aembed_documents(web_texts)

            # Calculate cosine similarities
            similarities = []
            for web_emb in web_embeddings:
                similarity = self._cosine_similarity(query_embedding, web_emb)
                similarities.append(similarity)

            # Return average similarity
            return float(np.mean(similarities)) if similarities else 0.0

        except Exception as e:
            return 0.5  # Default if calculation fails

    def _calculate_confidence_score(self, tool_type: str, **kwargs) -> float:
        """
        Calculate combined confidence score.
        For RAG: weighted blend of faithfulness + retrieval_quality
        For Web: weighted blend of consistency + relevance
        Returns: 0-1 score
        """
        if tool_type == "rag":
            faithfulness = kwargs.get("faithfulness", "partial")
            retrieval_quality = kwargs.get("retrieval_quality", 0.5)

            # Map faithfulness to score
            faithfulness_score = {
                "supported": 1.0,
                "partial": 0.5,
                "unsupported": 0.0,
                "error": 0.3
            }.get(faithfulness, 0.5)

            # Weighted combination: 60% faithfulness, 40% retrieval quality
            confidence = (0.6 * faithfulness_score) + (0.4 * retrieval_quality)

        elif tool_type == "web":
            consistency = kwargs.get("consistency", "partial")
            relevance = kwargs.get("relevance", 0.5)

            # Map consistency to score
            consistency_score = {
                "consistent": 1.0,
                "partial": 0.5,
                "unsupported": 0.0,
                "error": 0.3
            }.get(consistency, 0.5)

            # Weighted combination: 60% consistency, 40% relevance
            confidence = (0.6 * consistency_score) + (0.4 * relevance)

        else:
            confidence = 0.0

        return float(confidence)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

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

    def _format_web_context(self, web_results: List[Document]) -> str:
        """Format web search results for evaluation."""
        if not web_results:
            return "No web search results available."

        context_parts = []
        for i, doc in enumerate(web_results, 1):
            title = doc.metadata.get("title", "Untitled")
            url = doc.metadata.get("url", "")
            content = doc.page_content

            context_parts.append(
                f"[Web Result {i}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}\n"
            )

        return "\n".join(context_parts)