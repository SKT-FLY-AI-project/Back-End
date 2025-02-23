from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select as async_select
from sqlalchemy import desc
from datetime import datetime
import json

from app.database import get_db
from app.models import Conversation, Message
from app.schemas import ChatMessage
from app.utils.connection import manager
from app.services.conversation import get_or_create_conversation, save_message, get_conversation_history, get_user_conversations
from app.utils.llm_utils import generate_vts_response

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)

@router.post("/{userid}/create")
async def save(
    userid: str,
    image_url: str = Body(...),
    title: str = Body(...),
    rich_description: str = Body(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        # 기존 감상(대화) 조회 없이 무조건 새 레코드 생성
        new_conversation = Conversation(
            user_id=userid,
            image_url=image_url,
            title=title,
            rich_description=rich_description
        )
        db.add(new_conversation)
        await db.flush()
        await db.commit()
        await db.refresh(new_conversation)
        return {"conversation_id": new_conversation.id, "message": "감상이 저장되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# HTTP 엔드포인트: 대화 시작/계속
@router.post("/{userid}/chat")
async def start_vts_conversation(
    userid: str, 
    chat_data: ChatMessage = Body(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        request = chat_data.request
        image_url = chat_data.image_url
        title = chat_data.title
        rich_description = chat_data.rich_description
        conversation_id = chat_data.conversation_id
        
        # 대화 얻기 또는 생성
        conversation = await get_or_create_conversation(
            db, userid, image_url, title, rich_description, conversation_id
        )
        
        # 기존 대화 기록 불러오기
        conversation_history = await get_conversation_history(db, conversation.id)
        
        # 새로운 메시지 추가
        await save_message(db, conversation.id, "user", request)
        user_requests = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
        user_requests.append(request)
        
        # LLM을 사용한 응답 생성
        reaction, question = generate_vts_response(request, user_requests)
        response = reaction + '\n' + question
        
        # 응답을 대화 기록에 저장
        await save_message(db, conversation.id, "assistant", response)
        
        # 대화 최종 업데이트 시간 갱신
        conversation.updated_at = datetime.utcnow()
        await db.commit()
        
        return {
            "conversation_id": conversation.id,
            "response": response,
            "conversation": conversation_history + [
                {"role": "user", "content": request},
                {"role": "assistant", "content": response}
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# 대화 목록 조회 엔드포인트
@router.get("/{userid}")
async def get_user_conversation_list(
    userid: str,
    date: str = None,
    title: str = None,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    try:
        conversations = await get_user_conversations(db, userid, limit=limit)
        
        if date:
            try:
                filter_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="날짜 형식은 YYYY-MM-DD 여야 합니다.")
            conversations = [
                conv for conv in conversations 
                if conv.created_at.date() == filter_date
            ]
        
        if title:
            conversations = [
                conv for conv in conversations 
                if conv.title == title
            ]
            
        result = []
        
        for conv in conversations:
            # 각 대화의 마지막 메시지 조회
            query = async_select(Message).where(
                Message.conversation_id == conv.id
            ).order_by(desc(Message.created_at)).limit(10)
            
            last_message_result = await db.execute(query)
            last_message = last_message_result.scalars().first()
            
            result.append({
                "conversation_id": conv.id,
                "image_url": conv.image_url,
                "title": conv.title,
                # "name": conv.name,
                "rich_description": conv.rich_description,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "last_message": last_message.content if last_message else None,
                "last_message_role": last_message.role if last_message else None
            })
        
        print(result)
        return {"conversations": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# 특정 대화 히스토리 조회 엔드포인트
@router.get("/{conversation_id}/detail")
async def get_conversation_detail(
    conversation_id: str,
    userid: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        # 대화 정보 조회
        result = await db.execute(
            async_select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalars().first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다")
            
        if conversation.user_id != userid:
            raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
        
        # 메시지 히스토리 조회
        messages = await get_conversation_history(db, conversation_id)
        
        return {
            "conversation_id": conversation.id,
            "image_url": conversation.image_url,
            "title": conversation.title,
            "rich_description": conversation.rich_description,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "messages": messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")