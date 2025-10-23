from typing import List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from prompts.synthesizer_agent_prompt import SYNTHESIZER_AGENT_PROMPT
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes

class SynthesizerAgent:

    def __init__(self):
        self.llm = loadLLM(LLMType.SYNTHESIZER_AGENT)
        self.chain = SYNTHESIZER_AGENT_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            documents = state.get("retrieved_documents", [])
            web_results = state.get("web_search_results", [])
            confidence_score = state.get("confidence_score", 0.0)
            
            # Get reflection feedback from previous attempts
            synthesis_attempts = state.get("synthesis_attempts", 0)
            reflection_issues = state.get("reflection_issues", [])
            reflection_suggestions = state.get("reflection_suggestions", [])
            previous_quality_score = state.get("answer_quality_score", 0.0)

            rag_context = self._format_rag_context(documents)
            web_context = self._format_web_context(web_results)
            
            # Debug logging
            logger.info(f"[SynthesizerAgent] Documents: {len(documents)}, Web results: {len(web_results)}")
            logger.info(f"[SynthesizerAgent] Web context length: {len(web_context)}")
            if web_results:
                logger.info(f"[SynthesizerAgent] First web result: {web_results[0].metadata.get('title', 'No title')}")
            
            # Format reflection feedback for the prompt
            reflection_feedback = self._format_reflection_feedback(
                synthesis_attempts, reflection_issues, reflection_suggestions, previous_quality_score
            )

            response = await self.chain.ainvoke({
                "user_query": query,
                "rag_context": rag_context,
                "web_context": web_context,
                "confidence_score": f"{confidence_score:.2f}",
                "num_documents": len(documents),
                "used_web_search": "Yes" if web_results else "No",
                "reflection_feedback": reflection_feedback,
                "synthesis_attempts": synthesis_attempts,
            })

            sources = self._extract_sources(documents, web_results)

            return {
                **state,
                "response": response,
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"[SynthesizerAgent] Error during synthesis: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"SynthesizerAgent error: [{type(e).__name__}]: {str(e)}",
            )

    def _format_rag_context(self, documents: List[Document]) -> str:
        if not documents:
            return "No relevant documents found in knowledge base."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content

            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {Path(source).name}, Page {page}\n"
                f"Content: {content}\n"
            )

        return "\n".join(context_parts)

    def _format_web_context(self, documents: List[Document]) -> str:
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
        sources = []
        for doc in documents:
            source_path = doc.metadata.get("source", "Unknown")
            sources.append({
                "type": "document",
                "source": Path(source_path).name,
                "page": str(doc.metadata.get("page", "?")),
            })

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
        previous_quality_score: float
    ) -> str:
        """Format reflection feedback for the prompt."""
        if synthesis_attempts <= 1:
            return "This is the first synthesis attempt."
        
        feedback_parts = [
            f"This is synthesis attempt #{synthesis_attempts}.",
            f"Previous attempt quality score: {previous_quality_score:.2f}",
        ]
        
        if issues:
            feedback_parts.append("Issues identified in previous attempt:")
            for i, issue in enumerate(issues, 1):
                feedback_parts.append(f"  {i}. {issue}")
        
        if suggestions:
            feedback_parts.append("Suggestions for improvement:")
            for i, suggestion in enumerate(suggestions, 1):
                feedback_parts.append(f"  {i}. {suggestion}")
        
        return "\n".join(feedback_parts)
