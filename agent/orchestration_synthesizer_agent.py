from typing import List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from prompts.orchestration_synthesizer_agent_prompt import ORCHESTRATION_SYNTHESIZER_AGENT_PROMPT
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes


class OrchestrationSynthesizerAgent:
    """
    Synthesizer specialized for merging RAG + Web search results.

    This agent is used in the orchestration stage to create a comprehensive answer
    by combining:
    1. Knowledge from RAG (documents in knowledge base)
    2. Information from web search (current/external data)

    Key responsibilities:
    - Merge RAG and web search results intelligently
    - Prioritize authoritative sources
    - Handle conflicting information
    - Provide citations from both RAG and web
    - Incorporate reflection feedback for iterative improvement
    """

    def __init__(self):
        self.llm = loadLLM(LLMType.ORCHESTRATION_SYNTHESIZER_AGENT)
        self.chain = ORCHESTRATION_SYNTHESIZER_AGENT_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("refined_query") or state.get("user_query", "")
            documents = state.get("retrieved_documents", [])
            web_results = state.get("web_search_results", [])
            confidence_score = state.get("confidence_score", 0.0)

            # Get reflection feedback from previous orchestration attempts
            synthesis_attempts = state.get("synthesis_attempts", 0)
            reflection_issues = state.get("reflection_issues", [])
            reflection_suggestions = state.get("reflection_suggestions", [])
            previous_quality_score = state.get("answer_quality_score", 0.0)

            # Get RAG reflection reasoning to understand why RAG was insufficient
            rag_reflection_reasoning = state.get("rag_reflection_reasoning", "")

            logger.info(
                f"[Orchestration Synthesizer] Merging RAG ({len(documents)} docs) + "
                f"Web ({len(web_results)} results), attempt #{synthesis_attempts + 1}"
            )

            rag_context = self._format_rag_context(documents)
            web_context = self._format_web_context(web_results)

            # Format reflection feedback for iterative improvement
            reflection_feedback = self._format_reflection_feedback(
                synthesis_attempts, reflection_issues, reflection_suggestions,
                previous_quality_score, rag_reflection_reasoning
            )

            response = await self.chain.ainvoke({
                "user_query": query,
                "rag_context": rag_context,
                "web_context": web_context,
                "confidence_score": f"{confidence_score:.2f}",
                "num_documents": len(documents),
                "num_web_results": len(web_results),
                "reflection_feedback": reflection_feedback,
                "synthesis_attempts": synthesis_attempts + 1,
            })

            sources = self._extract_sources(documents, web_results)

            logger.info(
                f"[Orchestration Synthesizer] Generated merged answer "
                f"({len(response)} chars, {len(sources)} sources)"
            )

            return {
                **state,
                "response": response,
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"[Orchestration Synthesizer] Error during synthesis: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"Orchestration Synthesizer error: [{type(e).__name__}]: {str(e)}",
            )

    def _format_rag_context(self, documents: List[Document]) -> str:
        """Format RAG documents for the prompt."""
        if not documents:
            return "No documents from knowledge base (RAG was insufficient)."

        context_parts = ["RAG Documents (Knowledge Base):\n"]
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content
            score = doc.metadata.get("relevance_score", 0.0)

            context_parts.append(
                f"[RAG Doc {i}] (Relevance: {score:.3f})\n"
                f"Source: {Path(source).name}, Page {page}\n"
                f"Content: {content}\n"
            )

        return "\n".join(context_parts)

    def _format_web_context(self, documents: List[Document]) -> str:
        """Format web search results for the prompt."""
        if not documents:
            return "No web search results available."

        context_parts = ["Web Search Results:\n"]
        for i, doc in enumerate(documents, 1):
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

    def _extract_sources(
        self,
        documents: List[Document],
        web_results: List[Document]
    ) -> List[Dict[str, Any]]:
        """Extract sources from both RAG and web results."""
        sources = []

        # Add RAG document sources
        for doc in documents:
            source_path = doc.metadata.get("source", "Unknown")
            sources.append({
                "type": "document",
                "source": Path(source_path).name,
                "page": str(doc.metadata.get("page", "?")),
                "relevance_score": doc.metadata.get("relevance_score", 0.0),
            })

        # Add web search sources
        for doc in web_results:
            sources.append({
                "type": "web",
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("url", ""),
            })

        return sources

    def _format_reflection_feedback(
        self,
        synthesis_attempts: int,
        issues: List[str],
        suggestions: List[str],
        previous_quality_score: float,
        rag_reflection_reasoning: str
    ) -> str:
        """Format reflection feedback for iterative improvement."""
        if synthesis_attempts == 0:
            feedback_parts = [
                "This is the first orchestration synthesis attempt.",
                f"Why RAG was insufficient: {rag_reflection_reasoning}",
            ]
            return "\n".join(feedback_parts)

        feedback_parts = [
            f"This is orchestration synthesis attempt #{synthesis_attempts + 1}.",
            f"Previous attempt quality score: {previous_quality_score:.2f}",
            f"RAG insufficiency reason: {rag_reflection_reasoning}",
        ]

        if issues:
            feedback_parts.append("\nIssues identified in previous attempt:")
            for i, issue in enumerate(issues, 1):
                feedback_parts.append(f"  {i}. {issue}")

        if suggestions:
            feedback_parts.append("\nSuggestions for improvement:")
            for i, suggestion in enumerate(suggestions, 1):
                feedback_parts.append(f"  {i}. {suggestion}")

        return "\n".join(feedback_parts)
