from langchain_core.tools import tool
from pydantic import BaseModel, Field
from domain.enums.workflow_state import RoutingDecision, WorkflowState
from domain.enums.llm_type import LLMType
from infrastructure.llm.loader import loadLLM
from langchain.agents import create_agent
from prompts.rag_reflection_agent_prompt import RAG_REFLECTION_AGENT_PROMPT
from core.utils.exception import ArcFusionException
from domain.enums.error_code import ArcFusionErrorCodes

class RAGReflectionResult(BaseModel):
    routing_decision: str = Field(
        description="The routing decision: 'rag_sufficient' or 'rag_insufficient'"
    )
    comment: str = Field(
        description="A comment explaining why the routing decision was made to give feedback to the planner agent",
        default=""
    )
    
@tool(args_schema=RAGReflectionResult)
def finalize_rag_reflection(routing_decision: str, comment: str) -> str:
    """
    Validates and finalizes the routing decision after LLM analysis.
    This tool only validates the decision format - the LLM makes the intelligent choice.
    Use this tool to submit your final routing decision.

    Args:
        routing_decision: The final routing decision ('rag_sufficient', 'rag_insufficient').
        comment: A comment explaining why the routing decision was made to give feedback to the planner agent.

    Returns:
        str: JSON string containing the validated routing decision and comment.
    """
    import json
    
    cleaned_decision = routing_decision.strip().lower()
    valid_decisions = [RoutingDecision.RAG_SUFFICIENT.value, RoutingDecision.RAG_INSUFFICIENT.value]

    if cleaned_decision not in valid_decisions:
        result = RAGReflectionResult(
            routing_decision="invalid_decision",
            comment="Invalid routing decision"
        )
    else:
        result = RAGReflectionResult(
            routing_decision=cleaned_decision,
            comment=comment
        )
    
    return json.dumps(result.dict())

class RAGReflectionAgent:

    def __init__(self):
        self.llm = loadLLM(LLMType.RAG_REFLECTION_AGENT)
        self.tools = [finalize_rag_reflection]
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=RAG_REFLECTION_AGENT_PROMPT,
        )
        
    async def invoke(self, state: WorkflowState) -> WorkflowState:
        try:
            user_query = state.get("user_query", "")
            rag_synthesizer_response = state.get("rag_synthesizer_response", "")

            if not rag_synthesizer_response or rag_synthesizer_response.strip() == "":
                return {
                    **state,
                    "routing_decision": RoutingDecision.NOT_RELEVANT.value,
                    "rag_reflection_comment": "No relevant documents found in knowledge base. Please use web search to answer the question.",
                }

                f"[RAGReflectionAgent] Evaluating RAG sufficiency for query: {user_query[:100]}..."
            )

            # Step 1: Run the agent
            formatted_prompt = RAG_REFLECTION_AGENT_PROMPT.format(
                user_query=user_query,
                retrieved_summarized_answer=rag_synthesizer_response
            )
            agent_input = {"messages": [{"role": "user", "content": formatted_prompt}]}
            agent_response = await self.agent.ainvoke(agent_input)

            # Step 2: Extract messages and initialize decision
            messages = agent_response.get("messages", [])
            decision = None
            comment = ""

            # Step 3: Search messages (reverse order = latest first)
            # Look for the first valid tool result to avoid unnecessary processing
            for msg in reversed(messages):
                # 3.1 If it's a ToolMessage (result from finalize_rag_reflection tool)
                if msg.__class__.__name__ == "ToolMessage":
                    tool_content = getattr(msg, "content", None)
                    # Parse JSON response from tool
                    if tool_content:
                        try:
                            import json
                            if isinstance(tool_content, str):
                                # Handle empty string case
                                if not tool_content.strip():
                                    continue
                                tool_result = json.loads(tool_content)
                            else:
                                tool_result = tool_content
                            
                            decision = tool_result.get("routing_decision")
                            comment = tool_result.get("comment", "")
                            
                            # If we get a valid decision from tool, use it immediately
                            if decision and decision in [RoutingDecision.RAG_SUFFICIENT.value, RoutingDecision.RAG_INSUFFICIENT.value]:
                                return {
                                    **state,
                                    "routing_decision": decision,
                                    "rag_reflection_comment": comment,
                                }
                        except (json.JSONDecodeError, AttributeError) as e:
                # 3.2 If it's an AIMessage that triggered tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") == "finalize_rag_reflection":
                            args = tool_call.get("args", {})
                            decision = args.get("routing_decision")
                            comment = args.get("comment", "")
                            # If we have a valid decision from tool call, use it immediately
                            if decision and decision in [RoutingDecision.RAG_SUFFICIENT.value, RoutingDecision.RAG_INSUFFICIENT.value]:
                                return {
                                    **state,
                                    "routing_decision": decision,
                                    "rag_reflection_comment": comment,
                                }

            # Step 4: Fallback — try parsing last AI message text if no valid tool result found
            last_message = messages[-1] if messages else None
            if last_message and hasattr(last_message, "content"):
                decision = getattr(last_message, "content", "").strip()
            else:
                decision = RoutingDecision.RAG_INSUFFICIENT.value
            # Step 5: Validate and normalize decision
            if decision:
                decision = str(decision).strip().lower()
            if decision not in [
                RoutingDecision.RAG_SUFFICIENT.value,
                RoutingDecision.RAG_INSUFFICIENT.value,
            ]:
                decision = RoutingDecision.RAG_INSUFFICIENT.value

            # Step 6: Return updated state
            return {
                **state,
                "routing_decision": decision,
                "rag_reflection_comment": comment,
            }

        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"RAGReflectionAgent error: [{type(e).__name__}]: {str(e)}",
            )