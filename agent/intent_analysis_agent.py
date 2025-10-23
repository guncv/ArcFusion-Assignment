"""
Intent Analysis Agent: Analyzes user queries and provides RAG-augmented responses.

This agent:
1. Receives clear, refined queries from the clarification workflow
2. Determines if RAG/web search is needed
3. Retrieves relevant context
4. Generates comprehensive answers with citations
"""

from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from infrastructure.llm.loader import loadLLM
from domain.enums.llm_type import LLMType
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes


class IntentAnalysisAgent:
    """
    Analyze query intent and provide RAG-augmented responses.
    """

    def __init__(self):
        self.llm = loadLLM(LLMType.INTENT_ANALYSIS_AGENT)

        # Lazy load RAG pipeline (only when needed)
        self._rag_pipeline = None

        logger.info("[IntentAnalysisAgent] Initialized")

    @property
    def rag_pipeline(self):
        """Lazy load RAG pipeline to avoid circular imports."""
        if self._rag_pipeline is None:
            from infrastructure.rag import RAGPipeline
            self._rag_pipeline = RAGPipeline(
                collection_name="arcfusion_docs",
                use_web_search=True,
                top_k=5
            )
        return self._rag_pipeline

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        """
        Process query with intent analysis and RAG.

        Args:
            state: Workflow state with user query and session

        Returns:
            Updated state with generated response and sources
        """
        try:
            user_query = state.get("user_query", "")
            session_id = state.get("session_id", "")

            logger.info(f"[IntentAnalysisAgent] Processing query: '{user_query[:100]}...'")

            # Check if query needs RAG or web search
            needs_rag = self._should_use_rag(user_query)
            needs_web = self._should_use_web_search(user_query)

            logger.info(
                f"[IntentAnalysisAgent] Intent analysis: "
                f"needs_rag={needs_rag}, needs_web={needs_web}"
            )

            # Generate response using RAG pipeline
            result = await self.rag_pipeline.generate(
                query=user_query,
                session_id=session_id,
                use_web_search=needs_web,
                use_ensemble=True
            )

            # Format response with sources
            answer = result.get("answer", "I couldn't generate an answer.")
            sources = result.get("sources", [])

            # Add source citations if available
            if sources:
                formatted_answer = self._format_answer_with_sources(answer, sources)
            else:
                formatted_answer = answer

            logger.info(
                f"[IntentAnalysisAgent] Generated answer with {len(sources)} sources"
            )

            return {
                **state,
                "response": formatted_answer,
                "sources": sources,
                "used_rag": needs_rag,
                "used_web_search": result.get("used_web_search", False),
            }

        except Exception as e:
            logger.error(f"[IntentAnalysisAgent] Error during invoke: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"IntentAnalysisAgent error: [{type(e).__name__}]: {str(e)}",
            )

    def _should_use_rag(self, query: str) -> bool:
        """Determine if query should use RAG (knowledge base)."""
        rag_keywords = [
            "document", "paper", "according to", "in the",
            "what does", "explain", "how does", "describe"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in rag_keywords)

    def _should_use_web_search(self, query: str) -> bool:
        """Determine if query needs web search."""
        web_keywords = [
            "latest", "recent", "today", "yesterday", "this week",
            "current", "now", "2024", "2025", "news", "trends"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in web_keywords)

    def _format_answer_with_sources(self, answer: str, sources: list) -> str:
        """Format answer with source citations."""
        if not sources:
            return answer

        source_text = "\n\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            if source["type"] == "document":
                source_text += f"{i}. {source['source']} (Page {source['page']})\n"
            elif source["type"] == "web":
                source_text += f"{i}. [{source['title']}]({source['url']})\n"

        return answer + source_text
