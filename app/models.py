from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import BINARY
from datetime import datetime, timezone
import uuid

Base = declarative_base()

# User 테이블
class User(Base):
    __tablename__ = "user"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # UUID 사용
    name = Column(String(255), unique=True, nullable=False)
    password = Column(BINARY(60), nullable=False)
    difficulty_lv = Column(Enum('E', 'H'), nullable=True)
    conversations_st = Column(String(255), nullable=True)
    clr_knowledge = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # One-to-Many 관계: 한 사용자는 여러 Conversations(사진/설명)을 가짐.
    conversationss = relationship(
        "Conversations",
        back_populates="user",
        cascade="all, delete",
        primaryjoin="User.id == Conversations.user_id"
    )

# 데이터베이스 모델
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # 길이 36을 지정
    user_id = Column(String(255), index=True)
    photo_url = Column(String(2083), nullable=True)
    image_title = Column(String(255))
    vlm_description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # 길이 36을 지정
    conversation_id = Column(String(36), ForeignKey("conversations.id"))  # 길이 36을 지정
    role = Column(String(255))  # "user" 또는 "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")