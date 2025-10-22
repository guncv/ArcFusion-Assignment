import json
import os
from typing import Dict
from pathlib import Path

from core.config.config import nested_config as config
from core.log.logger import logger
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from domain.enums.llm_type import LLMType
from pydantic import Field

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

class PersistentChatHistory(InMemoryChatMessageHistory):
    """
    Chat history that automatically saves to disk.
    Extends InMemoryChatMessageHistory with automatic persistence.
    """
    
    session_id: str = Field(description="Unique session identifier")
    persist_dir: Path = Field(description="Directory to persist chat history")
    file_path: Path = Field(description="Full path to the session file")
    
    def __init__(self, session_id: str, persist_dir: str = "./data/chat_history"):
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        file_path = persist_path / f"{session_id}.json"
        
        super().__init__(
            session_id=session_id,
            persist_dir=persist_path,
            file_path=file_path
        )
        
        # Load existing history if it exists
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load chat history from disk if file exists"""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert dict messages back to BaseMessage objects
                for msg_data in data:
                    if msg_data['type'] == 'human':
                        self.add_message(HumanMessage(content=msg_data['content']))
                    elif msg_data['type'] == 'ai':
                        self.add_message(AIMessage(content=msg_data['content']))
                    elif msg_data['type'] == 'system':
                        self.add_message(SystemMessage(content=msg_data['content']))
                
                logger.info(f"Loaded {len(self.messages)} messages for session {self.session_id}")
                
            except Exception as e:
                logger.error(f"Error loading chat history for {self.session_id}: {e}")
    
    def _save_to_disk(self):
        """Save current chat history to disk"""
        try:
            # Convert messages to serializable format
            data = []
            for msg in self.messages:
                data.append({
                    'type': msg.__class__.__name__.lower().replace('message', ''),
                    'content': msg.content
                })
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving chat history for {self.session_id}: {e}")
    
    def add_message(self, message: BaseMessage):
        """Add message and auto-save"""
        super().add_message(message)
        self._save_to_disk()
    
    def add_user_message(self, message: str):
        """Add user message and auto-save"""
        super().add_user_message(message)
        self._save_to_disk()
    
    def add_ai_message(self, message: str):
        """Add AI message and auto-save"""
        super().add_ai_message(message)
        self._save_to_disk()
    
    def clear(self):
        """Clear history and remove file"""
        super().clear()
        if self.file_path.exists():
            try:
                self.file_path.unlink()
                logger.info(f"Cleared and removed history file for session {self.session_id}")
            except Exception as e:
                logger.error(f"Error removing history file for {self.session_id}: {e}")


_chat_histories: Dict[str, PersistentChatHistory] = {}

def getChatHistory(session_id: str) -> PersistentChatHistory:
    """
    Get chat history for a session with automatic persistence.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        PersistentChatHistory instance that auto-saves to disk
    """
    history = _chat_histories.get(session_id)
    if history is None:
        # Get persist directory from config or use default
        persist_dir = config.get('chat_history', {}).get('persist_dir', './data/chat_history')
        history = PersistentChatHistory(session_id, persist_dir)
        _chat_histories[session_id] = history
        logger.info(f"Created persistent chat history for session: {session_id}")
    return history

def clearChatHistory(session_id: str) -> bool:
    """
    Clear chat history for a session (removes from memory and disk).
    
    Args:
        session_id: Session to clear
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        history = _chat_histories.get(session_id)
        if history is not None:
            history.clear()
        _chat_histories.pop(session_id, None)
        logger.info(f"Cleared chat history for session: {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error clearing chat history for session {session_id}: {e}")
        return False

llm = LLM()
def loadLLM(type: LLMType):
    return llm.loadLLM(type.value)
