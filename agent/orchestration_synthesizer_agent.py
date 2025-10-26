import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from domain.enums.workflow_state import WorkflowState, ToolType
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from prompts.orchestration_synthesizer_agent_prompt import (
    CURRENT_RAG_SYNTHESIZER_PROMPT,
    CURRENT_WEB_SYNTHESIZER_PROMPT,
    MERGED_SYNTHESIZER_PROMPT
)
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes
from infrastructure.evaluation.evaluation_service import EvaluationService
import logging

logger = logging.getLogger(__name__)

class OrchestrationSynthesizerAgent:
    def __init__(self):
        self.evaluation_service = EvaluationService()
        self.llm = loadLLM(LLMType.ORCHESTRATION_SYNTHESIZER_AGENT)
        self.rag_chain = CURRENT_RAG_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()
        self.web_chain = CURRENT_WEB_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()
        self.merged_chain = MERGED_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            documents = state.get("rag_synthesizer_response", "")
            web_results = state.get("web_search_results", [])
            rag_documents = state.get("retrieved_documents_with_scores", [])
            old_response = state.get("response", "")
            selected_tool = state.get("selected_tool", "")

            rag_context = self._format_rag_context(documents)
            web_context = self._format_web_context(web_results)
            rag_docs_context = self._format_rag_documents_context(rag_documents)

            # Generate current iteration response based on selected tool
            if selected_tool == ToolType.RAG_SEARCH.value:
                current_response = await self.rag_chain.ainvoke({
                    "user_query": query,
                    "rag_context": rag_context,
                    "rag_docs_context": rag_docs_context,
                })
            elif selected_tool == ToolType.WEB_SEARCH.value:
                current_response = await self.web_chain.ainvoke({
                    "user_query": query,
                    "web_context": web_context,
                })
            else:
                # Fallback: if no tool selected, return empty current response
                current_response = ""

            # Generate merged response (old_response + current_response)
            merged_response = await self.merged_chain.ainvoke({
                "user_query": query,
                "old_response": old_response,
                "current_response": current_response,
            })

            # Prepare updated state
            updated_state = {
                **state,
                "current_synthesized_response": current_response,
                "response": merged_response,
            }

            # Trigger background evaluation (fire-and-forget, non-blocking)
            try:
                asyncio.create_task(self.evaluation_service.evaluate_and_save(updated_state))
                logger.info("Background evaluation triggered successfully")
            except Exception as eval_error:
                # Don't fail synthesis if evaluation trigger fails
                logger.error(f"Failed to trigger evaluation: {eval_error}")

            return updated_state

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"Orchestration Synthesizer error: [{type(e).__name__}]: {str(e)}",
            )

    def _format_rag_context(self, documents: str) -> str:
        if not documents:
            return "No relevant documents found in knowledge base."

        return f"RAG Documents (Knowledge Base): {documents}\n"

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

    def _format_rag_documents_context(self, rag_documents: List) -> str:
        if not rag_documents:
            return "No RAG documents retrieved."

        context_parts = ["RAG Retrieved Documents:\n"]
        for i, doc_item in enumerate(rag_documents, 1):
            if isinstance(doc_item, dict) and "document" in doc_item:
                # Handle RetrievedDocument format with score
                doc = doc_item["document"]
                score = doc_item.get("score", 0.0)
                content = doc.page_content
                metadata = doc.metadata
                title = metadata.get("title", "Untitled")
                
                context_parts.append(
                    f"[RAG Document {i}] (Score: {score:.3f})\n"
                    f"Title: {title}\n"
                    f"Content: {content}\n"
                )
            else:
                # Handle direct Document format
                content = doc_item.page_content if hasattr(doc_item, 'page_content') else str(doc_item)
                metadata = doc_item.metadata if hasattr(doc_item, 'metadata') else {}
                title = metadata.get("title", "Untitled")
                
                context_parts.append(
                    f"[RAG Document {i}]\n"
                    f"Title: {title}\n"
                    f"Content: {content}\n"
                )

        return "\n".join(context_parts)