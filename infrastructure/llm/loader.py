from typing import Any, Dict

from core.config.config import nested_config as config
from core.log.logger import logger
from domain.enums.llm_type import LLMType
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

class LLM:
    def __init__(self):
        self.model = None
        self.provider = None
        self.temperature = None
        self.api_key = None

    def loadLLM(self, type: str):
        self.model = config[type]["model"]
        self.provider = config[type]["api_provider"]
        self.temperature = config[type]["temperature"]
        self.api_key = config[type]["api_key"]
        self.max_tokens = config[type]["max_tokens"]
        
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                openai_api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                anthropic_api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError("Unsupported LLM provider")

def _get_chat_history_config() -> Dict[str, Any]:
    return config.get("chat_history", {})

def _build_redis_url(redis_config: Dict[str, Any]) -> str:
    host = redis_config.get("host", "localhost")
    port = redis_config.get("port", 6379)
    db = redis_config.get("db", 0)
    password = redis_config.get("password")

    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"

def getChatHistory(session_id: str) -> BaseChatMessageHistory:
    chat_config = _get_chat_history_config()
    redis_config = chat_config.get("redis", {})

    key_prefix = redis_config.get("key_prefix", "chat_history:")
    ttl = chat_config.get("ttl")

    return RedisChatMessageHistory(
        session_id=session_id,
        url=_build_redis_url(redis_config),
        key_prefix=key_prefix,
        ttl=ttl,
    )

def clearChatHistory(session_id: str) -> bool:
    try:
        history = getChatHistory(session_id)
        history.clear()
        return True
    except Exception as e:
        logger.error(f"Error clearing chat history for session {session_id}: {e}")
        return False

llm = LLM()

def loadLLM(type: LLMType):
    return llm.loadLLM(type.value)

def loadLLMWithHistory(type: LLMType):
    base_llm = llm.loadLLM(type.value)
    
    chain_with_history = RunnableWithMessageHistory(
        base_llm,
        getChatHistory,
        input_messages_key="input",
        history_messages_key="history"
    )

    return chain_with_history
