from sqlalchemy import Column, String, DateTime, ForeignKey, Enum, Boolean, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from app.database import Base

# User 테이블
class User(Base):
    __tablename__ = "user"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=True)  # 필요시 BINARY 타입으로 변경 가능
    difficulty_lv = Column(Enum('E', 'H'), nullable=True)
    description_st = Column(String(255), nullable=True)
    clr_knowledge = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                        onupdate=lambda: datetime.now(timezone.utc))

    # 한 사용자는 여러 Conversation을 가짐 (양방향 관계)
    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete",
        primaryjoin="User.id == Conversation.user_id"
    )

# Conversation 테이블
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("user.id"), index=True)
    image_url = Column(String(2083), nullable=True)
    title = Column(String(255))
    artist = Column(String(255))
    rich_description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                        onupdate=lambda: datetime.now(timezone.utc))
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    # 역방향 관계: Conversation -> User
    user = relationship("User", back_populates="conversations")

# Message 테이블
class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id"))
    role = Column(String(255))  # 예: "user" 또는 "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # 역방향 관계: Message -> Conversation
    conversation = relationship("Conversation", back_populates="messages")