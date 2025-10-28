from langchain_core.output_parsers import StrOutputParser
from src.graph import WorkflowState, ToolType
from src.constants import LLMAgentName, LLMType
from src.infras import llm_loader
from src.prompts import (
    CURRENT_RAG_SYNTHESIZER_PROMPT,
    CURRENT_WEB_SYNTHESIZER_PROMPT,
    COMBINED_RAG_WEB_SYNTHESIZER_PROMPT,
    MERGED_SYNTHESIZER_PROMPT
)
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from src.agent.base import AgentInterface
from src.utils.format import (
    format_web_context,
    format_rag_documents_context
)

class SynthesizerAgent(AgentInterface):
    def __init__(self):
        self._evaluation_service = None
        self.llm = llm_loader.loadLLM(LLMType.SYNTHESIZER_AGENT)
        self.rag_chain = CURRENT_RAG_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()
        self.web_chain = CURRENT_WEB_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()
        self.combined_chain = COMBINED_RAG_WEB_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()
        self.merged_chain = MERGED_SYNTHESIZER_PROMPT | self.llm | StrOutputParser()
        self._agent_name = LLMAgentName.SYNTHESIZER_AGENT.value

    @property
    def name(self) -> str:
        return self._agent_name or super().name
    
    @property
    def evaluation_service(self):
        if self._evaluation_service is None:
            from src.services.evaluation import get_evaluation_service
            self._evaluation_service = get_evaluation_service()
        return self._evaluation_service

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        try:
            query = state.get("user_query", "")
            web_results = state.get("web_search_results", [])
            rag_documents = state.get("retrieved_documents_with_scores", [])
            selected_tool = state.get("selected_tool", "")

            web_context = format_web_context(web_results)
            rag_docs_context = format_rag_documents_context(rag_documents)

            # Check if we have BOTH RAG and Web results
            has_rag = bool(rag_documents)
            has_web = bool(web_results)

            # Generate current iteration response
            if has_rag and has_web:
                # Use combined synthesizer when both sources available
                current_response = await self.combined_chain.ainvoke({
                    "user_query": query,
                    "rag_context": rag_docs_context,
                    "web_context": web_context,
                })
            elif selected_tool == ToolType.RAG_SEARCH.value:
                current_response = await self.rag_chain.ainvoke({
                    "user_query": query,
                    "rag_context": rag_docs_context,
                })
            elif selected_tool == ToolType.WEB_SEARCH.value:
                current_response = await self.web_chain.ainvoke({
                    "user_query": query,
                    "web_context": web_context,
                })
            else:
                current_response = ""

            # Synthesis completed - MetaAssessor will evaluate quality
            updated_state = {
                **state,
                "response": current_response,
            }

            return updated_state

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )