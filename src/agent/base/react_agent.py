from typing import List, Optional, Dict, Any
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from src.graph import WorkflowState
from src.constants import LLMType
from src.infras import llm_loader
from src.repositories.chat_history import get_chat_history_repository
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes
from .agent_interface import AgentInterface
from src.infras.log import logger

class ReActAgent(AgentInterface):

    def __init__(
        self,
        llm_type: LLMType,
        system_prompt: str,
        tools: List[BaseTool],
        agent_name: Optional[str] = None
    ):
        self.llm_type = llm_type
        self.llm = llm_loader.loadLLM(llm_type)
        self.system_prompt = system_prompt
        self.tools = tools
        self._agent_name = agent_name

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    @property
    def name(self) -> str:
        return self._agent_name or super().name

    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        # Process the workflow state using the ReAct agent.
        try:
            # Subclasses should override this with specific implementation
            raise NotImplementedError(
                f"{self.name} must implement the invoke method"
            )
        except Exception as e:
            raise ArcFusionException(
                error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
            )

    async def get_chat_history(self, state: WorkflowState):
        session_id = state.get("session_id", "unknown")
        logger.debug(f"[{session_id}] {self.name} requesting chat history")
        repo = get_chat_history_repository()
        messages = await repo.get_messages(session_id)
        return messages

    async def build_message_list(
        self,
        user_query: str,
        include_history: bool = False,
        state: Optional[WorkflowState] = None
    ) -> List[Dict[str, str]]:
        messages = []

        if include_history and state:
            session_id = state.get("session_id", "unknown")
            messages_list = await self.get_chat_history(state)
            if messages_list:
                logger.info(f"[{session_id}] {self.name} including {len(messages_list)} chat history messages in context")
            for msg in messages_list:
                # Handle ChatMessage objects from database
                if hasattr(msg, 'message_type'):
                    if msg.message_type == 'human':
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.message_type == 'ai':
                        messages.append({"role": "assistant", "content": msg.content})
                # Handle LangChain message objects
                elif hasattr(msg, 'type'):
                    if msg.type == 'human':
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.type == 'ai':
                        messages.append({"role": "assistant", "content": msg.content})

        # Add current query
        messages.append({"role": "user", "content": user_query})

        return messages

    async def ainvoke_agent(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        # Invoke the ReAct agent with messages.
        return await self.agent.ainvoke({"messages": messages})

    def extract_tool_result(
        self,
        messages: List,
        tool_name: str,
        valid_values: Optional[List[str]] = None,
        result_key: str = "content"
    ) -> Optional[str]:
        for msg in reversed(messages):
            # Check ToolMessage (result from tool execution)
            if msg.__class__.__name__ == "ToolMessage":
                content = getattr(msg, result_key, None)
                if content:
                    # Handle JSON results
                    if isinstance(content, str):
                        try:
                            import json
                            if content.strip() and content.strip().startswith('{'):
                                content = json.loads(content)
                        except (json.JSONDecodeError, ValueError):
                            pass

                    # Extract value
                    if isinstance(content, dict):
                        value = content.get(result_key, content)
                    else:
                        value = content

                    # Validate if needed
                    if valid_values:
                        if isinstance(value, str) and value.strip().lower() in valid_values:
                            return value.strip().lower()
                    else:
                        return value

            # Check AIMessage with tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.get("name") == tool_name:
                        args = tool_call.get("args", {})
                        value = args.get("decision") if "decision" in args else args.get(result_key)

                        if valid_values:
                            if isinstance(value, str) and value.strip().lower() in valid_values:
                                return value.strip().lower()
                        else:
                            return value

        return None