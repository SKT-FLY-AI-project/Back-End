from typing import Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select as async_select
from sqlalchemy import desc
from app.models import Conversation, Message

# 대화 생성 또는 조회
async def get_or_create_conversation(
    db: AsyncSession, user_id: str, image_url: str, title: str, 
    rich_description: Optional[str] = None, 
    conversation_id: Optional[str] = None
):
    if conversation_id:
        # 기존 대화 조회
        result = await db.execute(
            async_select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalars().first()
        if conversation and conversation.user_id == user_id:
            return conversation
    
    # 새 대화 생성 또는 이미지 제목으로 최근 대화 조회
    if not conversation_id:
        result = await db.execute(
            async_select(Conversation)
            .where(Conversation.user_id == user_id, Conversation.title == title)
            .order_by(desc(Conversation.updated_at))
            .limit(1)
        )
        existing_conversation = result.scalars().first()
        if existing_conversation:
            return existing_conversation
    # 새 대화 생성
    new_conversation = Conversation(
        user_id=user_id,
        image_url=image_url,
        title=title,
        rich_description=rich_description
    )
    db.add(new_conversation)
    await db.commit()
    await db.refresh(new_conversation)
    return new_conversation

# 메시지 저장
async def save_message(db: AsyncSession, conversation_id: str, role: str, content: str):
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content
    )
    db.add(message)
    await db.commit()
    return message

# 대화 히스토리 조회
async def get_conversation_history(db: AsyncSession, conversation_id: str):
    result = await db.execute(
        async_select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    return [{"role": msg.role, "content": msg.content} for msg in messages]

# 사용자의 모든 대화 조회
async def get_user_conversations(db: AsyncSession, user_id: str, date: str = None, limit: int = 10):
    # 기본 쿼리: user_id 조건만 추가
    query = async_select(Conversation).where(Conversation.user_id == user_id)
    
    query = query.order_by(desc(Conversation.updated_at)).limit(limit)
    
    result = await db.execute(query)
    conversations = result.scalars().all()
    return conversations