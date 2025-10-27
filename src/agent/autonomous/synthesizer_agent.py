import asyncio
from dotenv.main import logger
from langchain_core.output_parsers import StrOutputParser
from src.graph import WorkflowState, ToolType
from src.constants import LLMAgentName, LLMType
from src.infras import llm_loader
from src.prompts import (
    CURRENT_RAG_SYNTHESIZER_PROMPT,
    CURRENT_WEB_SYNTHESIZER_PROMPT,
    MERGED_SYNTHESIZER_PROMPT
)
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import AgentInterface
from src.utils.format import (
    format_rag_context,
    format_web_context,
    format_rag_documents_context
)

class SynthesizerAgent(AgentInterface):
    def __init__(self):
        self._evaluation_service = None
        self.llm = llm_loader.loadLLM(LLMType.SYNTHESIZER_AGENT)
        self.rag_chain = CURRENT_RAG_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()
        self.web_chain = CURRENT_WEB_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()
        self.merged_chain = MERGED_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()
        self._agent_name = LLMAgentName.SYNTHESIZER_AGENT.value
    
    @property
    def evaluation_service(self):
        if self._evaluation_service is None:
            from src.services.evaluation_service import get_evaluation_service
            self._evaluation_service = get_evaluation_service()
        return self._evaluation_service

    @property
    def name(self) -> str:
        return self._agent_name or super().name

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            documents = state.get("rag_synthesizer_response", "")
            web_results = state.get("web_search_results", [])
            rag_documents = state.get("retrieved_documents_with_scores", [])
            old_response = state.get("response", "")
            selected_tool = state.get("selected_tool", "")

            rag_context = format_rag_context(documents)
            web_context = format_web_context(web_results)
            rag_docs_context = format_rag_documents_context(rag_documents)

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
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )