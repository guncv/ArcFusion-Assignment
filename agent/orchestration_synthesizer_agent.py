from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from domain.enums.workflow_state import WorkflowState
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from prompts.orchestration_synthesizer_agent_prompt import ORCHESTRATION_SYNTHESIZER_AGENT_PROMPT
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes

class OrchestrationSynthesizerAgent:
    def __init__(self):
        self.llm = loadLLM(LLMType.ORCHESTRATION_SYNTHESIZER_AGENT)
        self.chain = ORCHESTRATION_SYNTHESIZER_AGENT_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            documents = state.get("rag_synthesizer_response", "")
            web_results = state.get("web_search_results", [])
            rag_documents = state.get("retrieved_documents_with_scores", [])
            old_response = state.get("response", "")
            
            rag_context = self._format_rag_context(documents)
            web_context = self._format_web_context(web_results)
            rag_docs_context = self._format_rag_documents_context(rag_documents)

            response = await self.chain.ainvoke({
                "user_query": query,
                "rag_context": rag_context,
                "web_context": web_context,
                "rag_docs_context": rag_docs_context,
                "old_response": old_response,
            })

            return {
                **state,
                "response": response,
            }

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