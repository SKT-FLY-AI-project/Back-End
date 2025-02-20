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


# # Conversations 테이블 (사진/설명, 부모: One)
# class Conversations(Base):
#     __tablename__ = "conversations"
#     id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # 길이 36을 지정
#     user_id = Column(String(255),  ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
#     photo_url = Column(String(2083), nullable=True)
#     image_title = Column(String(255))
#     vlm_conversations = Column(Text, nullable=True)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
#     # Many-to-One 관계: 하나의 Conversations은 하나의 User에 속함.
#     user = relationship(
#         "User",
#         back_populates="conversationss",
#         primaryjoin="User.id == Conversations.user_id"
#     )
    
#     # One-to-Many 관계: 하나의 Conversations은 여러 Message(채팅 기록)을 가짐.
#     messages = relationship(
#         "Message",
#         back_populates="conversations",
#         cascade="all, delete-orphan"
#     )



# # Message 테이블 (자식: Many)
# class Message(Base):
#     __tablename__ = "messages"
    
#     id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
#     # 외래키 이름를 conversations_id로 변경하여 Conversations.id를 참조
#     conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
#     role = Column(String(255))  # "user" 또는 "assistant"
#     content = Column(Text)
#     created_at = Column(DateTime, default=datetime.utcnow)
    
#     # Many-to-One 관계: 하나의 Message는 하나의 Conversations에 속함.
#     conversations = relationship(
#         "Conversations",
#         back_populates="messages",
#         primaryjoin="Conversations.id == Message.conversation_id"
#     )

# Conversations 테이블 (사진/설명, 부모: One)
class Conversation(Base): # 변경1
    __tablename__ = "conversations"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # 길이 36을 지정
    user_id = Column(String(255),  ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    photo_url = Column(String(2083), nullable=True)
    image_title = Column(String(255))
    vlm_conversations = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Many-to-One 관계: 하나의 Conversations은 하나의 User에 속함.
    user = relationship(
        "User",
        back_populates="conversationss",
        primaryjoin="User.id == Conversations.user_id"
    )
    
    # One-to-Many 관계: 하나의 Conversations은 여러 Message(채팅 기록)을 가짐.
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )



# Message 테이블 (자식: Many)
class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(255))  # "user" 또는 "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Many-to-One 관계: 하나의 Message는 하나의 Conversations에 속함.
    conversation = relationship(
        "Conversation",
        back_populates="messages",
        primaryjoin="Conversations.id == Message.conversation_id"
    )