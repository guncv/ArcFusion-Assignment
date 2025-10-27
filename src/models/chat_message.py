from sqlalchemy import Column, String, Float, Text, DateTime, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), nullable=False, index=True)
    message_type = Column(String(50), nullable=False)  # 'human' or 'ai'
    content = Column(Text, nullable=False)
    sequence = Column(Integer, nullable=False)  # Order within session
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<ChatMessage(session={self.session_id}, type={self.message_type}, seq={self.sequence})>"
