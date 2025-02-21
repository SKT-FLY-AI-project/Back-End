from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select as async_select
from sqlalchemy import desc
from datetime import datetime
import json

from database import get_db
from app.models import Conversation, Message
from app.schemas import ChatMessage
from app.utils.connection import manager
from app.services.conversation import get_or_create_conversation, save_message, get_conversation_history, get_user_conversations
from app.utils.llm_utils import generate_vts_response

router = APIRouter()

@router.post("/save/{userid}")
async def save(
    userid: str,
    photo_url: str = Body(...),
    image_title: str = Body(...),
    vlm_description: str = Body(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        # 기존 감상(대화) 조회 없이 무조건 새 레코드 생성
        new_conversation = Conversation(
            user_id=userid,
            photo_url=photo_url,
            image_title=image_title,
            vlm_description=vlm_description
        )
        db.add(new_conversation)
        await db.flush()
        print("제발")
        await db.commit()
        await db.refresh(new_conversation)
        return {"conversation_id": new_conversation.id, "message": "감상이 저장되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# HTTP 엔드포인트: 대화 시작/계속
@router.post("/bot/{userid}")
async def start_vts_conversation(
    userid: str, 
    chat_data: ChatMessage = Body(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        request = chat_data.request
        photo_url = chat_data.photo_url
        image_title = chat_data.image_title
        vlm_description = chat_data.vlm_description
        conversation_id = chat_data.conversation_id
        
        # 대화 얻기 또는 생성
        conversation = await get_or_create_conversation(
            db, userid, photo_url, image_title, vlm_description, conversation_id
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
@router.get("/conversations/{userid}")
async def get_user_conversation_list(
    userid: str,
    date: str = None,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    try:
        conversations = await get_user_conversations(db, userid, limit)
        result = []
        
        for conv in conversations:
            # 각 대화의 마지막 메시지 조회
            query = async_select(Message).where(
                Message.conversation_id == conv.id
            ).order_by(desc(Message.created_at)).limit(limit)
            
            last_message_result = await db.execute(query)
            last_message = last_message_result.scalars().first()
            
            result.append({
                "conversation_id": conv.id,
                "photo_url": conv.photo_url,
                "image_title": conv.image_title,
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
@router.get("/conversation/{conversation_id}")
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
            "photo_url": conversation.photo_url,
            "image_title": conversation.image_title,
            "vlm_description": conversation.vlm_description,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "messages": messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# 웹소켓 연결 엔드포인트
@router.websocket("/ws/chat/{userid}")
async def websocket_endpoint(websocket: WebSocket, userid: str):
    await manager.connect(websocket, userid)
    try:
        while True:
            try:
                # 클라이언트로부터 메시지 수신
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # 메시지 타입에 따른 처리
                if message_data.get("message_type") == "ping":
                    await manager.send_message(userid, {"message_type": "pong"})
                    continue
                
                # DB 세션 생성
                async with get_db() as db:
                    # 채팅 메시지 처리
                    request = message_data.get("request", "")
                    photo_url = message_data.get("photo_url")
                    image_title = message_data.get("image_title", "unknown")
                    vlm_description = message_data.get("vlm_description")
                    conversation_id = message_data.get("conversation_id")
                    
                    # 대화 얻기 또는 생성
                    conversation = await get_or_create_conversation(
                        db, userid, photo_url, image_title, vlm_description, conversation_id
                    )
                    
                    # 기존 대화 기록 불러오기
                    conversation_history = await get_conversation_history(db, conversation.id)
                    
                    # 새로운 메시지 추가
                    await save_message(db, conversation.id, "user", request)
                    user_requests = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
                    user_requests.append(request)
                    
                    # LLM으로 응답 생성
                    reaction, question = generate_vts_response(request, user_requests)
                    response = reaction + '\n' + question
                    
                    # 응답을 대화 기록에 저장
                    await save_message(db, conversation.id, "assistant", response)
                    
                    # 대화 최종 업데이트 시간 갱신
                    conversation.updated_at = datetime.utcnow()
                    await db.commit()
                    
                    # 응답 전송
                    await manager.send_message(userid, {
                        "message_type": "chat_response",
                        "conversation_id": conversation.id,
                        # "session_id": session_id,
                        "response": response,
                    })
                    print("message sent")
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "유효하지 않은 JSON 형식입니다."}))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"메시지 처리 중 오류 발생: {str(e)}"}))
                
    except WebSocketDisconnect:
        manager.disconnect(userid)
    except Exception as e:
        print(f"웹소켓 오류: {str(e)}")
        manager.disconnect(userid)