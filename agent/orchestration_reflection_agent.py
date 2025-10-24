from typing import List, Dict, Any
from pydantic import BaseModel, Field
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from prompts.reflection_agent_prompt import REFLECTION_AGENT_PROMPT
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes


class ReflectionResult(BaseModel):
    """Structured output for reflection evaluation."""
    is_sufficient: bool = Field(
        description="Whether the answer is sufficient and meets quality standards"
    )
    quality_score: float = Field(
        description="Overall quality score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of specific issues if answer is insufficient"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improving the answer if retrying"
    )


class OrchestrationReflectionAgent:
    """
    Agent responsible for evaluating the final answer quality after orchestration (RAG + Web).

    This agent acts as the final quality gate, determining if the generated answer:
    - Fully addresses the user's query with RAG + Web results
    - Has proper citations and evidence from both sources
    - Is clear, coherent, and well-structured
    - Meets quality standards

    If the answer is insufficient, it triggers replanning with improvement suggestions.
    """

    def __init__(self, max_synthesis_attempts: int = 3):
        """
        Initialize orchestration reflection agent.

        Args:
            max_synthesis_attempts: Maximum number of planning attempts (replanning loops) before giving up
        """
        self.llm = loadLLM(LLMType.ORCHESTRATION_REFLECTION_AGENT)
        self.max_planning_attempts = max_synthesis_attempts  # Renamed conceptually, but keeping param name for backward compatibility

        # Create structured output LLM with tool calling
        self.structured_llm = self.llm.bind_tools(
            tools=[self._create_reflection_tool()],
            tool_choice="finalize_reflection"
        )

    def _create_reflection_tool(self) -> Dict[str, Any]:
        """Create the reflection evaluation tool schema."""
        return {
            "type": "function",
            "function": {
                "name": "finalize_reflection",
                "description": "Finalize the reflection evaluation with quality assessment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_sufficient": {
                            "type": "boolean",
                            "description": "Whether the answer is sufficient and meets quality standards"
                        },
                        "quality_score": {
                            "type": "number",
                            "description": "Overall quality score from 0.0 to 1.0",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "issues": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of specific issues if answer is insufficient"
                        },
                        "suggestions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Suggestions for improving the answer if retrying"
                        }
                    },
                    "required": ["is_sufficient", "quality_score"]
                }
            }
        }

    async def invoke(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate the synthesized answer quality.

        Args:
            state: Current workflow state with response and sources

        Returns:
            Updated state with reflection results and retry decision
        """
        try:
            user_query = state.get("refined_query") or state.get("user_query", "")
            generated_answer = state.get("response", "")
            sources = state.get("sources", [])
            planning_attempts = state.get("planning_attempts", 1)

            logger.info(
                f"[OrchestrationReflectionAgent] Evaluating answer quality "
                f"(planning attempt {planning_attempts}/{self.max_planning_attempts})"
            )

            # Format sources for display
            sources_text = self._format_sources(sources)

            # Determine quality threshold based on attempt number
            quality_threshold = 0.6 if planning_attempts == 1 else 0.7

            # Invoke LLM with structured output
            response = await self.structured_llm.ainvoke(
                REFLECTION_AGENT_PROMPT.format_messages(
                    user_query=user_query,
                    generated_answer=generated_answer,
                    sources=sources_text,
                    synthesis_attempts=planning_attempts,
                    max_attempts=self.max_planning_attempts,
                    quality_threshold=quality_threshold
                )
            )

            # Extract tool call result
            reflection_result = self._extract_reflection_result(response)

            # Enforce max attempts - don't retry if we've hit the limit
            if planning_attempts >= self.max_planning_attempts:
                logger.warning(
                    f"[OrchestrationReflectionAgent] Max planning attempts ({self.max_planning_attempts}) reached. "
                    f"Accepting current answer with quality score: {reflection_result.quality_score:.2f}"
                )
                reflection_result.is_sufficient = True
                reflection_result.issues.append(
                    "Maximum planning attempts reached - accepting current answer"
                )

            logger.info(
                f"[OrchestrationReflectionAgent] Evaluation complete: "
                f"{'SUFFICIENT' if reflection_result.is_sufficient else 'INSUFFICIENT'} "
                f"(quality: {reflection_result.quality_score:.2f})"
            )

            if not reflection_result.is_sufficient:
                logger.info(
                    f"[OrchestrationReflectionAgent] Issues found: {', '.join(reflection_result.issues)}"
                )
                logger.info(
                    f"[OrchestrationReflectionAgent] Suggestions: {', '.join(reflection_result.suggestions)}"
                )

            return {
                **state,
                "is_answer_sufficient": reflection_result.is_sufficient,
                "answer_quality_score": reflection_result.quality_score,
                "reflection_issues": reflection_result.issues,
                "reflection_suggestions": reflection_result.suggestions,
            }

        except Exception as e:
            logger.error(f"[OrchestrationReflectionAgent] Error during reflection: {e}", exc_info=True)

            # Default to accepting the answer if reflection fails
            logger.warning("[OrchestrationReflectionAgent] Reflection failed, accepting current answer")
            return {
                **state,
                "is_answer_sufficient": True,
                "answer_quality_score": 0.5,
                "reflection_issues": [f"Reflection error: {str(e)}"],
                "reflection_suggestions": [],
            }

    def _extract_reflection_result(self, response: Any) -> ReflectionResult:
        """Extract reflection result from LLM response with tool calls."""
        try:
            # Check if response has tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call = response.tool_calls[0]
                args = tool_call.get("args", {})

                return ReflectionResult(
                    is_sufficient=args.get("is_sufficient", True),
                    quality_score=args.get("quality_score", 0.7),
                    issues=args.get("issues", []),
                    suggestions=args.get("suggestions", [])
                )

            # Fallback: Accept the answer if no tool call found
            logger.warning("[OrchestrationReflectionAgent] No tool call found in response, accepting answer")
            return ReflectionResult(
                is_sufficient=True,
                quality_score=0.7,
                issues=[],
                suggestions=[]
            )

        except Exception as e:
            logger.error(f"[OrchestrationReflectionAgent] Error extracting result: {e}", exc_info=True)
            # Default to accepting on error
            return ReflectionResult(
                is_sufficient=True,
                quality_score=0.5,
                issues=[f"Extraction error: {str(e)}"],
                suggestions=[]
            )

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for display in prompt."""
        if not sources:
            return "No sources available"

        source_lines = []
        for i, source in enumerate(sources, 1):
            if source.get("type") == "document":
                source_lines.append(
                    f"{i}. Document: {source.get('source', 'Unknown')}, "
                    f"Page {source.get('page', '?')}"
                )
            elif source.get("type") == "web":
                source_lines.append(
                    f"{i}. Web: {source.get('title', 'Untitled')} - "
                    f"{source.get('url', '')}"
                )

        return "\n".join(source_lines)
