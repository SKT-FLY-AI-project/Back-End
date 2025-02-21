import json
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select as async_select
from sqlalchemy import desc
from models import Conversation, Message
from groq import Groq

# Groq 클라이언트 초기화
client = Groq(api_key="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")

async def get_or_create_conversation(
    db: AsyncSession, 
    user_id: str, 
    title: str, 
    rich_description: str = None, 
    conversation_id: str = None
):
    if conversation_id:
        result = await db.execute(
            async_select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalars().first()
        if conversation and conversation.user_id == user_id:
            return conversation

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

    new_conversation = Conversation(
        user_id=user_id,
        title=title,
        rich_description=rich_description
    )
    db.add(new_conversation)
    await db.commit()
    await db.refresh(new_conversation)
    return new_conversation

async def save_message(
    db: AsyncSession, 
    conversation_id: str, 
    role: str, 
    content: str
):
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content
    )
    db.add(message)
    await db.commit()
    return message

async def get_conversation_history(
    db: AsyncSession, 
    conversation_id: str
):
    result = await db.execute(
        async_select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    return [{"role": msg.role, "content": msg.content} for msg in messages]

async def get_user_conversations(
    db: AsyncSession, 
    user_id: str, 
    limit: int = 10
):
    result = await db.execute(
        async_select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    conversations = result.scalars().all()
    return conversations

def generate_vts_response(user_input, conversation_history):
    """
    사용자의 입력과 대화 히스토리를 기반으로 적절한 반응과 질문을 생성하는 함수.
    """
    context = "\n".join(conversation_history[-3:])  # 최근 3개만 유지
    prompt = f"""
    사용자가 미술 작품을 감상하고 있습니다.
    이전 대화:
    {context}

    사용자의 입력:
    "{user_input}"

    AI의 역할:
    1. 사용자의 감상에 대해 적절한 반응을 제공합니다.
    2. 새로운 질문을 생성하여 자연스럽게 대화를 이어갑니다.

    AI의 응답 형식:
    1. 반응: (사용자의 감상을 반영한 피드백)
    2. 질문: (VTS 기반의 적절한 추가 질문)
    """
    completion = client.chat.completions.create(
        model="qwen-2.5-coder-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=150,
        top_p=0.95
    )
    response = completion.choices[0].message.content.strip()
    try:
        response_parts = response.split("\n")
        reaction = response_parts[0].strip() if response_parts else "흥미로운 생각이에요."
        question = response_parts[1].strip() if len(response_parts) > 1 else "이 작품을 보고 어떤 점이 가장 인상적이었나요?"
    except:
        reaction, question = response, "이 작품을 보고 어떤 점이 가장 인상적이었나요?"
    return reaction[7:], question[7:]
