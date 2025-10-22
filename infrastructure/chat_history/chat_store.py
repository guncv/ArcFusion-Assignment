import redis
from typing import Optional
from pathlib import Path
import json
import os

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import FileChatMessageHistory

from core.config.config import nested_config as config
from core.log.logger import logger


class ChatHistoryStore:

    def __init__(self):
        self.backend = config.get("chat_history", {}).get("backend", "redis")
        self.ttl = config.get("chat_history", {}).get("ttl", 3600)
        self._redis_client: Optional[redis.Redis] = None
        self._initialize_backend()

    def _initialize_backend(self):
        if self.backend == "redis":
            try:
                redis_config = config.get("chat_history", {}).get("redis", {})
                self._redis_client = redis.Redis(
                    host=redis_config.get("host", "localhost"),
                    port=int(redis_config.get("port", 6379)),
                    db=int(redis_config.get("db", 0)),
                    password=redis_config.get("password") or None,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                
                self._redis_client.ping()
            except Exception as e:
                logger.error(f"[ChatHistoryStore] Failed to connect to Redis: {e}")
                logger.warning("[ChatHistoryStore] Falling back to file-based storage")
                self.backend = "file"

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if self.backend == "redis" and self._redis_client:
            return self._get_redis_history(session_id)
        else:
            return self._get_file_history(session_id)

    def _get_redis_history(self, session_id: str) -> RedisChatMessageHistory:
        redis_config = config.get("chat_history", {}).get("redis", {})
        key_prefix = redis_config.get("key_prefix", "chat_history:")

        history = RedisChatMessageHistory(
            session_id=session_id,
            url=self._get_redis_url(),
            key_prefix=key_prefix,
            ttl=self.ttl,
        )

        logger.debug(f"[ChatHistoryStore] Retrieved Redis history for session: {session_id}")
        return history

    def _get_file_history(self, session_id: str) -> FileChatMessageHistory:
        """Get file-based chat history for a session."""
        file_config = config.get("chat_history", {}).get("file", {})
        base_path = file_config.get("base_path", "./chat_history")
        
        # Ensure directory exists
        os.makedirs(base_path, exist_ok=True)
        
        # Create file path for this session
        file_path = os.path.join(base_path, f"{session_id}.json")
        
        history = FileChatMessageHistory(file_path=file_path)
        
        logger.debug(f"[ChatHistoryStore] Retrieved file history for session: {session_id}")
        return history

    def _get_redis_url(self) -> str:
        """Get Redis URL from configuration."""
        redis_config = config.get("chat_history", {}).get("redis", {})
        
        host = redis_config.get("host", "localhost")
        port = redis_config.get("port", 6379)
        password = redis_config.get("password")
        db = redis_config.get("db", 0)
        
        # Build Redis URL
        if password:
            url = f"redis://:{password}@{host}:{port}/{db}"
        else:
            url = f"redis://{host}:{port}/{db}"
            
        return url

    def clear_session(self, session_id: str) -> bool:
        try:
            history = self.get_session_history(session_id)
            history.clear()
            return True
        except Exception as e:
            logger.error(f"[ChatHistoryStore] Error clearing session {session_id}: {e}")
            return False

    def add_message(self, session_id: str, message: BaseMessage):
        try:
            history = self.get_session_history(session_id)
            history.add_message(message)
            logger.debug(f"[ChatHistoryStore] Added {message.__class__.__name__} to session {session_id}")
        except Exception as e:
            logger.error(f"[ChatHistoryStore] Error adding message to session {session_id}: {e}")

    def add_user_message(self, session_id: str, content: str):
        self.add_message(session_id, HumanMessage(content=content))

    def add_ai_message(self, session_id: str, content: str):
        self.add_message(session_id, AIMessage(content=content))

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> list[BaseMessage]:
        try:
            history = self.get_session_history(session_id)
            messages = history.messages

            if limit and len(messages) > limit:
                messages = messages[-limit:]

            return messages
        except Exception as e:
            logger.error(f"[ChatHistoryStore] Error getting messages for session {session_id}: {e}")
            return []

_chat_store: Optional[ChatHistoryStore] = None

def get_chat_store() -> ChatHistoryStore:
    global _chat_store
    if _chat_store is None:
        _chat_store = ChatHistoryStore()
    return _chat_store

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = get_chat_store()
    return store.get_session_history(session_id)


def clear_session_history(session_id: str) -> bool:
    store = get_chat_store()
    return store.clear_session(session_id)
