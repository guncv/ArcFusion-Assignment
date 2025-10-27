from sqlalchemy import Column, String, Float, Text, DateTime, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime
Base = declarative_base()

class EvaluationMetrics(Base):
    __tablename__ = "evaluation_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), nullable=False, index=True)
    user_query = Column(Text, nullable=False)
    tool_type = Column(String(50), nullable=False, index=True)  # 'rag_search' or 'web_search'

    # RAG-specific metrics
    faithfulness = Column(String(50), nullable=True)  # 'supported' | 'partial' | 'unsupported'
    retrieval_quality = Column(Float, nullable=True)  # 0-1 score

    # WebSearch-specific metrics
    factual_consistency = Column(String(50), nullable=True)  # 'consistent' | 'partial' | 'unsupported'
    relevance_score = Column(Float, nullable=True)  # 0-1 score

    # Combined metrics (for both)
    confidence_score = Column(Float, nullable=False)  # 0-1 combined score

    # Response data
    current_response = Column(Text, nullable=True)  # The synthesized response

    # Additional metadata (JSONB for flexibility)
    additional_metadata = Column("metadata", JSON, nullable=True)  # Extra info like num_documents, etc.

    # Timestamp
    created_at = Column(DateTime, default=datetime.now, nullable=False, index=True)

    def __repr__(self):
        return f"<EvaluationMetrics(id={self.id}, tool={self.tool_type}, confidence={self.confidence_score})>"
