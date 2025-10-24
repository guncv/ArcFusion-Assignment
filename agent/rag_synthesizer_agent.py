from typing import List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from prompts.rag_synthesizer_agent_prompt import RAG_SYNTHESIZER_AGENT_PROMPT
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes


class RAGSynthesizerAgent:
    """
    Synthesizer specialized for RAG-only results.

    This agent generates answers using ONLY retrieved documents from the knowledge base.
    It does NOT use web search results - that's handled by OrchestrationSynthesizerAgent.

    Key responsibilities:
    - Synthesize answer from RAG documents only
    - Ensure all claims are supported by retrieved documents
    - Provide document citations
    - Stay within the scope of available knowledge
    """

    def __init__(self):
        self.llm = loadLLM(LLMType.RAG_SYNTHESIZER_AGENT)
        self.chain = RAG_SYNTHESIZER_AGENT_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("refined_query") or state.get("user_query", "")
            documents = state.get("retrieved_documents", [])
            confidence_score = state.get("confidence_score", 0.0)

            logger.info(
                f"[RAG Synthesizer] Synthesizing from {len(documents)} documents "
                f"(confidence: {confidence_score:.2f})"
            )

            rag_context = self._format_rag_context(documents)

            response = await self.chain.ainvoke({
                "user_query": query,
                "rag_context": rag_context,
                "confidence_score": f"{confidence_score:.2f}",
                "num_documents": len(documents),
            })

            sources = self._extract_sources(documents)

            logger.info(f"[RAG Synthesizer] Generated answer ({len(response)} chars)")

            return {
                **state,
                "response": response,
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"[RAG Synthesizer] Error during synthesis: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"RAG Synthesizer error: [{type(e).__name__}]: {str(e)}",
            )

    def _format_rag_context(self, documents: List[Document]) -> str:
        """Format RAG documents for the prompt."""
        if not documents:
            return "No relevant documents found in knowledge base."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content
            score = doc.metadata.get("relevance_score", 0.0)

            context_parts.append(
                f"[Document {i}] (Relevance: {score:.3f})\n"
                f"Source: {Path(source).name}, Page {page}\n"
                f"Content: {content}\n"
            )

        return "\n".join(context_parts)

    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source metadata from RAG documents."""
        sources = []
        for doc in documents:
            source_path = doc.metadata.get("source", "Unknown")
            sources.append({
                "type": "document",
                "source": Path(source_path).name,
                "page": str(doc.metadata.get("page", "?")),
                "relevance_score": doc.metadata.get("relevance_score", 0.0),
            })

        return sources
