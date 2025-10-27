from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.chat_message import ChatMessage
from src.infras.database.postgres import postgres_database
from src.infras.log import logger

class ChatHistoryRepository:
    def __init__(self):
        self.db = postgres_database

    async def add_message(self, session_id: str, message_type: str, content: str) -> Optional[ChatMessage]:
        try:
            async with self.db.async_session_factory() as session:
                result = await session.execute(
                    select(ChatMessage.sequence)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.sequence.desc())
                    .limit(1)
                )
                last_sequence = result.scalar()
                next_sequence = (last_sequence + 1) if last_sequence is not None else 1

                chat_message = ChatMessage(
                    session_id=session_id,
                    message_type=message_type,
                    content=content,
                    sequence=next_sequence
                )

                session.add(chat_message)
                await session.commit()
                await session.refresh(chat_message)

                logger.debug(f"Added message to session {session_id}: {message_type}")
                return chat_message

        except Exception as e:
            logger.error(f"Failed to add chat message: {e}")
            return None

    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ChatMessage]:
        try:
            async with self.db.async_session_factory() as session:
                query = (
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.sequence)
                    .offset(offset)
                )

                if limit:
                    query = query.limit(limit)

                result = await session.execute(query)
                messages = result.scalars().all()

                return list(messages)

        except Exception as e:
            logger.error(f"Failed to retrieve chat messages: {e}")
            return []

    async def get_latest_messages(
        self,
        session_id: str,
        count: int = 10
    ) -> List[ChatMessage]:
        try:
            async with self.db.async_session_factory() as session:
                result = await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.sequence.desc())
                    .limit(count)
                )
                messages = result.scalars().all()

                return list(reversed(messages))

        except Exception as e:
            logger.error(f"Failed to retrieve latest messages: {e}")
            return []

    async def clear_session(self, session_id: str) -> bool:
        """
        Delete all messages for a session.

        Args:
            session_id: The conversation session ID

        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.db.async_session_factory() as session:
                await session.execute(
                    delete(ChatMessage).where(ChatMessage.session_id == session_id)
                )
                await session.commit()

                logger.info(f"Cleared chat history for session {session_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to clear chat messages: {e}")
            return False

_chat_history_repo: Optional[ChatHistoryRepository] = None

def get_chat_history_repository() -> ChatHistoryRepository:
    global _chat_history_repo
    if _chat_history_repo is None:
        _chat_history_repo = ChatHistoryRepository()
    return _chat_history_repo
