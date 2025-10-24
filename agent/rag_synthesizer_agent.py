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

    def __init__(self):
        self.llm = loadLLM(LLMType.RAG_SYNTHESIZER_AGENT)
        self.chain = RAG_SYNTHESIZER_AGENT_PROMPT | self.llm | StrOutputParser()

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            retrieved_documents_with_scores = state.get("retrieved_documents_with_scores", [])

            # Check if there are any relevant documents
            if not retrieved_documents_with_scores or len(retrieved_documents_with_scores) == 0:
                logger.info(f"[RAG Synthesizer] No relevant documents found for query: {query[:100]}...")
                return {
                    **state,
                    "rag_synthesizer_response": "",
                }

            response = await self.chain.ainvoke({
                "user_query": query,
                "rag_context": retrieved_documents_with_scores,
                "num_documents": len(retrieved_documents_with_scores),
            })

            if any(phrase in response.lower() for phrase in [
                "does not contain", "not found", "not available", "missing", 
                "lacks information", "no information", "cannot find", "unable to find",
                "not provided", "not included", "not discussed", "not covered",
                "not addressed", "not specified", "not detailed", "not explained",
                "not described", "not mentioned", "not stated", "not indicated",
                "not shown", "not revealed", "not disclosed", "not presented",
                "not outlined", "not summarized", "web search may be needed",
                "additional information may be required", "further research may be necessary"
            ]):
                logger.info(f"[RAG Synthesizer] Detected feedback response, returning empty")
                response = ""

            return {
                **state,
                "rag_synthesizer_response": response,
            }

        except Exception as e:
            logger.error(f"[RAG Synthesizer] Error during synthesis: {e}", exc_info=True)
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"RAG Synthesizer error: [{type(e).__name__}]: {str(e)}",
            )