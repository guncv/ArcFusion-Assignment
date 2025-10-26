from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, message_to_dict, messages_from_dict
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from infrastructure.database.connection import db_connection
from infrastructure.database.models import ChatMessage
import asyncio
import logging

logger = logging.getLogger(__name__)


class PostgresChatMessageHistory(BaseChatMessageHistory):
    """
    PostgreSQL-based chat message history implementation.

    Implements LangChain's BaseChatMessageHistory interface using PostgreSQL.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._messages: List[BaseMessage] = []
        self._loaded = False

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from PostgreSQL."""
        if not self._loaded:
            # Load messages synchronously (LangChain expects sync interface)
            self._messages = asyncio.run(self._load_messages())
            self._loaded = True
        return self._messages

    async def _load_messages(self) -> List[BaseMessage]:
        """Load messages from database asynchronously."""
        try:
            async with db_connection.async_session_factory() as session:
                # Query messages for this session, ordered by sequence
                result = await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == self.session_id)
                    .order_by(ChatMessage.sequence)
                )
                chat_messages = result.scalars().all()

                # Convert to LangChain messages
                messages = []
                for msg in chat_messages:
                    if msg.message_type == "human":
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.message_type == "ai":
                        messages.append(AIMessage(content=msg.content))

                return messages

        except Exception as e:
            logger.error(f"Failed to load chat messages: {e}")
            return []

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history."""
        # Add to in-memory cache
        self._messages.append(message)

        # Save to database
        asyncio.run(self._save_message(message))

    async def _save_message(self, message: BaseMessage) -> None:
        """Save message to database asynchronously."""
        try:
            async with db_connection.async_session_factory() as session:
                # Determine message type
                if isinstance(message, HumanMessage):
                    message_type = "human"
                elif isinstance(message, AIMessage):
                    message_type = "ai"
                else:
                    message_type = "system"

                # Get next sequence number
                result = await session.execute(
                    select(ChatMessage.sequence)
                    .where(ChatMessage.session_id == self.session_id)
                    .order_by(ChatMessage.sequence.desc())
                    .limit(1)
                )
                last_sequence = result.scalar()
                next_sequence = (last_sequence + 1) if last_sequence is not None else 1

                # Create and save message
                chat_message = ChatMessage(
                    session_id=self.session_id,
                    message_type=message_type,
                    content=message.content,
                    sequence=next_sequence
                )

                session.add(chat_message)
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to save chat message: {e}")

    def clear(self) -> None:
        """Clear all messages for this session."""
        # Clear in-memory cache
        self._messages = []
        self._loaded = True

        # Delete from database
        asyncio.run(self._clear_messages())

    async def _clear_messages(self) -> None:
        """Delete all messages for this session from database."""
        try:
            async with db_connection.async_session_factory() as session:
                await session.execute(
                    delete(ChatMessage).where(ChatMessage.session_id == self.session_id)
                )
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to clear chat messages: {e}")