from typing import Dict, Any
from pydantic import BaseModel, Field
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from prompts.rag_reflection_agent_prompt import RAG_REFLECTION_AGENT_PROMPT
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes


class RAGReflectionResult(BaseModel):
    """Structured output for RAG sufficiency evaluation."""
    is_sufficient: bool = Field(
        description="Whether RAG results alone are sufficient to answer the query"
    )
    quality_score: float = Field(
        description="Quality score of RAG results from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Explanation for why RAG is or isn't sufficient"
    )
    needs_web_search: bool = Field(
        description="Whether web search is needed to supplement RAG results"
    )


class RAGReflectionAgent:
    """
    Agent responsible for evaluating if RAG retrieval results are sufficient.

    This agent acts as the first quality gate, determining if RAG alone can answer
    the user's query or if additional information from web search is needed.

    Decision logic:
    - If RAG results fully answer the query → is_sufficient=True, END workflow
    - If RAG results are incomplete/outdated/not found → is_sufficient=False, trigger orchestration
    """

    def __init__(self):
        """Initialize RAG reflection agent."""
        self.llm = loadLLM(LLMType.RAG_REFLECTION_AGENT)

        # Create structured output LLM with tool calling
        self.structured_llm = self.llm.bind_tools(
            tools=[self._create_rag_reflection_tool()],
            tool_choice="finalize_rag_reflection"
        )

    def _create_rag_reflection_tool(self) -> Dict[str, Any]:
        """Create the RAG reflection evaluation tool schema."""
        return {
            "type": "function",
            "function": {
                "name": "finalize_rag_reflection",
                "description": "Finalize the RAG sufficiency evaluation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_sufficient": {
                            "type": "boolean",
                            "description": "Whether RAG results alone are sufficient to answer the query"
                        },
                        "quality_score": {
                            "type": "number",
                            "description": "Quality score of RAG results from 0.0 to 1.0",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation for why RAG is or isn't sufficient"
                        },
                        "needs_web_search": {
                            "type": "boolean",
                            "description": "Whether web search is needed to supplement RAG results"
                        }
                    },
                    "required": ["is_sufficient", "quality_score", "reasoning", "needs_web_search"]
                }
            }
        }

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate if RAG results are sufficient to answer the user query.

        Args:
            state: Current workflow state with RAG results and synthesized response

        Returns:
            Updated state with RAG reflection results
        """
        try:
            user_query = state.get("refined_query") or state.get("user_query", "")
            generated_answer = state.get("response", "")
            retrieved_documents = state.get("retrieved_documents", [])
            confidence_score = state.get("confidence_score", 0.0)

            logger.info(
                f"[RAGReflectionAgent] Evaluating RAG sufficiency: "
                f"{len(retrieved_documents)} docs, confidence: {confidence_score:.2f}"
            )

            # Format documents for display
            documents_text = self._format_documents(retrieved_documents)

            # Invoke LLM with structured output
            response = await self.structured_llm.ainvoke(
                RAG_REFLECTION_AGENT_PROMPT.format_messages(
                    user_query=user_query,
                    generated_answer=generated_answer,
                    retrieved_documents=documents_text,
                    document_count=len(retrieved_documents),
                    confidence_score=confidence_score
                )
            )

            # Extract tool call result
            reflection_result = self._extract_reflection_result(response)

            logger.info(
                f"[RAGReflectionAgent] Evaluation complete: "
                f"{'SUFFICIENT' if reflection_result.is_sufficient else 'INSUFFICIENT'} "
                f"(quality: {reflection_result.quality_score:.2f})"
            )
            logger.info(f"[RAGReflectionAgent] Reasoning: {reflection_result.reasoning}")

            if reflection_result.needs_web_search:
                logger.info("[RAGReflectionAgent] Web search recommended - moving to orchestration")

            return {
                **state,
                "is_rag_sufficient": reflection_result.is_sufficient,
                "rag_quality_score": reflection_result.quality_score,
                "rag_reflection_reasoning": reflection_result.reasoning,
                "needs_orchestration": reflection_result.needs_web_search,
            }

        except Exception as e:
            logger.error(f"[RAGReflectionAgent] Error during reflection: {e}", exc_info=True)

            # Default to moving to orchestration if reflection fails
            logger.warning("[RAGReflectionAgent] Reflection failed, defaulting to orchestration")
            return {
                **state,
                "is_rag_sufficient": False,
                "rag_quality_score": 0.5,
                "rag_reflection_reasoning": f"Reflection error: {str(e)}",
                "needs_orchestration": True,
            }

    def _extract_reflection_result(self, response: Any) -> RAGReflectionResult:
        """Extract reflection result from LLM response with tool calls."""
        try:
            # Check if response has tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call = response.tool_calls[0]
                args = tool_call.get("args", {})

                return RAGReflectionResult(
                    is_sufficient=args.get("is_sufficient", False),
                    quality_score=args.get("quality_score", 0.5),
                    reasoning=args.get("reasoning", "No reasoning provided"),
                    needs_web_search=args.get("needs_web_search", True)
                )

            # Fallback: Move to orchestration if no tool call found
            logger.warning("[RAGReflectionAgent] No tool call found in response, defaulting to orchestration")
            return RAGReflectionResult(
                is_sufficient=False,
                quality_score=0.5,
                reasoning="No structured response received",
                needs_web_search=True
            )

        except Exception as e:
            logger.error(f"[RAGReflectionAgent] Error extracting result: {e}", exc_info=True)
            # Default to orchestration on error
            return RAGReflectionResult(
                is_sufficient=False,
                quality_score=0.3,
                reasoning=f"Extraction error: {str(e)}",
                needs_web_search=True
            )

    def _format_documents(self, documents: list) -> str:
        """Format retrieved documents for display in prompt."""
        if not documents:
            return "No documents retrieved from RAG"

        doc_lines = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}

            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', '?')
            score = metadata.get('relevance_score', 0.0)

            doc_lines.append(
                f"{i}. [Score: {score:.3f}] {source} (page {page})\n"
                f"   Content: {content[:200]}..."
            )

        return "\n\n".join(doc_lines)
