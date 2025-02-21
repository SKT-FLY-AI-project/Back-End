from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Body, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import json
import asyncio
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, ForeignKey, select, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.future import select as async_select
from sqlalchemy import select, func, desc
from datetime import datetime, timedelta
import os
from groq import Groq
import re
# from models import Base, Conversation, Message

router = APIRouter()
# router = APIRouter(prefix="/chat", tags=["Chatbot"])

os.environ['HF_HOME'] = "D:/huggingface_models"
client = Groq(api_key="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")

# SQLAlchemy 설정
DATABASE_URL = "mysql+aiomysql://root:1234@localhost/mydatabase"
# 프로덕션 환경에서는 PostgreSQL 추천: "postgresql+asyncpg://user:password@localhost/dbname"

async_engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)

Base = declarative_base()

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

# 클라이언트 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception:
                self.disconnect(client_id)
    
    def is_connected(self, client_id: str) -> bool:
        return client_id in self.active_connections

manager = ConnectionManager()

# 모델 정의
class ChatMessage(BaseModel):
    request: str
    photo_url: str
    image_title: str
    vlm_description: Optional[str] = None
    dominant_colors: Optional[List[List[int]]] = None
    conversation_id: Optional[str] = None  # 기존 대화를 이어가기 위한 ID

# DB 의존성
async def get_db():
    async with async_session() as session:
        yield session

# 데이터베이스 초기화 함수
async def init_database():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# 대화 생성 또는 조회
async def get_or_create_conversation(
    db: AsyncSession, user_id: str, photo_url: str, image_title: str, 
    vlm_description: Optional[str] = None, 
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
            .where(Conversation.user_id == user_id, Conversation.image_title == image_title)
            .order_by(desc(Conversation.updated_at))
            .limit(1)
        )
        existing_conversation = result.scalars().first()
        if existing_conversation:
            return existing_conversation
    # 새 대화 생성
    new_conversation = Conversation(
        user_id=user_id,
        photo_url=photo_url,
        image_title=image_title,
        vlm_description=vlm_description
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
# async def get_user_conversations(
#     user_id: str,
#     date: str = None,
#     limit: int = 10,
#     db: AsyncSession = Depends(get_db)
# ):
#     # 기본 쿼리: user_id 조건 추가
#     query = select(Conversation).where(Conversation.user_id == user_id)
    
#     # date 파라미터가 전달된 경우, 해당 날짜에 생성된 데이터만 필터링
#     if date:
#         try:
#             # "YYYY-MM-DD" 형식의 문자열을 date 객체로 변환
#             target_date = datetime.strptime(date, "YYYY-MM-DD").date()
#         except ValueError:
#             raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
#         # created_at 컬럼의 날짜 부분만 추출하여 비교
#         query = query.where(func.date(Conversation.created_at) == target_date)
    
#     # 최신 업데이트 순으로 정렬 및 limit 적용
#     query = query.order_by(desc(Conversation.updated_at)).limit(limit)
    
#     result = await db.execute(query)
#     conversations = result.scalars().all()
#     return conversations

async def get_user_conversations(db: AsyncSession, user_id: str, date: str = None, limit: int = 10):
    # 기본 쿼리: user_id 조건만 추가
    query = async_select(Conversation).where(Conversation.user_id == user_id)
    
    query = query.order_by(desc(Conversation.updated_at)).limit(limit)
    
    result = await db.execute(query)
    conversations = result.scalars().all()
    return conversations

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
                async with async_session() as db:
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

def generate_vts_response(user_input, conversation_history):
    """
    사용자의 입력과 대화 히스토리를 기반으로 적절한 반응과 질문을 생성하는 함수.
    """
    # 🔹 대화 맥락 정리
    context = "\n".join(conversation_history[-3:])  # 최근 3개만 유지 (메모리 최적화)

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
    
    # 🔹 응답을 "반응 + 질문"으로 분리
    try:
        response_parts = response.split("\n")
        reaction = response_parts[0].strip() if response_parts else "흥미로운 생각이에요."
        question = response_parts[1].strip() if len(response_parts) > 1 else "이 작품을 보고 어떤 점이 가장 인상적이었나요?"
    except:
        reaction, question = response, "이 작품을 보고 어떤 점이 가장 인상적이었나요?"

    return reaction[7:], question[7:]

# 앱 시작 시 백그라운드 태스크 시작하는 함수
def init_app(app):
    @app.on_event("startup")
    async def startup_event():
        await init_database()
        # asyncio.create_task(cleanup_sessions())