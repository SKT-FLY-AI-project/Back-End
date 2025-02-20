from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Body, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import json
import asyncio
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Text, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.future import select as async_select
import os
from groq import Groq

router = APIRouter()
# 라우터를 사용할 때 필요한 prefix나 tags를 추가할 수 있습니다.
os.environ['HF_HOME'] = "D:/huggingface_models"
client = Groq(api_key="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")

# SQLAlchemy 비동기 엔진 설정 (MySQL)
DATABASE_URL = "mysql+aiomysql://root:1234@localhost/mydatabase"
async_engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)

Base = declarative_base()

# ── 데이터베이스 모델 ──
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), index=True)
    image_title = Column(String(255))
    vlm_description = Column(Text, nullable=True)
    # 대화 내용을 JSON 문자열로 저장 (예: [{"role": "user", "content": "...", "timestamp": "..."}, ...])
    conversation_log = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 클라이언트 연결 관리 (WebSocket)
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

# ── Pydantic 모델 ──
class ChatMessage(BaseModel):
    request: str
    image_title: str
    vlm_description: Optional[str] = None
    dominant_colors: Optional[List[List[int]]] = None
    conversation_id: Optional[str] = None  # 기존 대화를 이어가기 위한 ID

class SessionRequest(BaseModel):
    session_id: str

# ── 메모리 캐시 (옵션) ──
conversation_sessions: Dict[str, Any] = {}

# ── DB 의존성 ──
async def get_db():
    async with async_session() as session:
        yield session

# ── 데이터베이스 초기화 함수 ──
async def init_database():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ── 대화 생성 또는 조회 ──
async def get_or_create_conversation(
    db: AsyncSession, 
    user_id: str, 
    image_title: str, 
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
    
    # 새 대화 생성: conversation_log 초기값은 빈 리스트(JSON)
    new_conversation = Conversation(
        user_id=user_id,
        image_title=image_title,
        vlm_description=vlm_description,
        conversation_log=json.dumps([])
    )
    db.add(new_conversation)
    await db.commit()
    await db.refresh(new_conversation)
    return new_conversation

# ── 대화 로그 업데이트 함수 ──
async def append_to_conversation_log(
    db: AsyncSession, 
    conversation: Conversation, 
    role: str, 
    content: str
):
    # 기존 대화 로그 불러오기 (JSON 파싱)
    if conversation.conversation_log:
        try:
            log = json.loads(conversation.conversation_log)
        except json.JSONDecodeError:
            log = []
    else:
        log = []
    # 새로운 메시지 추가
    log.append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    })
    conversation.conversation_log = json.dumps(log)
    conversation.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(conversation)
    return log

# ── 대화 히스토리 조회 ──
async def get_conversation_history(db: AsyncSession, conversation_id: str):
    result = await db.execute(
        async_select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalars().first()
    if conversation and conversation.conversation_log:
        try:
            return json.loads(conversation.conversation_log)
        except json.JSONDecodeError:
            return []
    return []

# ── 사용자의 모든 대화 조회 ──
async def get_user_conversations(db: AsyncSession, user_id: str, limit: int = 10):
    result = await db.execute(
        async_select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    conversations = result.scalars().all()
    return conversations

# ── HTTP 엔드포인트: 대화 시작/계속 ──
@router.post("/bot/{userid}")
async def start_vts_conversation(
    userid: str, 
    chat_data: ChatMessage = Body(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        request = chat_data.request
        image_title = chat_data.image_title
        vlm_description = chat_data.vlm_description
        conversation_id = chat_data.conversation_id
        
        # 대화 얻기 또는 생성
        conversation = await get_or_create_conversation(
            db, userid, image_title, vlm_description, conversation_id
        )
        
        # 기존 대화 기록 불러오기
        conversation_history = await get_conversation_history(db, conversation.id)
        
        # 사용자 메시지 저장
        await append_to_conversation_log(db, conversation, "user", request)
        conversation_history.append({"role": "user", "content": request})
        
        # LLM 호출을 위한 사용자 요청 리스트 생성 (모든 사용자 메시지 추출)
        user_requests = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
        user_requests.append(request)
        
        # LLM을 사용한 응답 생성 (예시 함수 사용)
        reaction, question = generate_vts_response(request, user_requests)
        response = reaction + '\n' + question
        
        # 어시스턴트 응답 저장
        await append_to_conversation_log(db, conversation, "assistant", response)
        conversation_history.append({"role": "assistant", "content": response})
        
        return {
            "conversation_id": conversation.id,
            "response": response,
            "conversation": conversation_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# ── 대화 목록 조회 엔드포인트 ──
@router.get("/conversations/{userid}")
async def get_user_conversation_list(
    userid: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    try:
        conversations = await get_user_conversations(db, userid, limit)
        result = []
        for conv in conversations:
            history = []
            if conv.conversation_log:
                try:
                    history = json.loads(conv.conversation_log)
                except json.JSONDecodeError:
                    history = []
            last_message = history[-1] if history else None
            result.append({
                "conversation_id": conv.id,
                "image_title": conv.image_title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "last_message": last_message["content"] if last_message else None,
                "last_message_role": last_message["role"] if last_message else None
            })
        return {"conversations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# ── 특정 대화 히스토리 조회 엔드포인트 ──
@router.get("/conversation/{conversation_id}")
async def get_conversation_detail(
    conversation_id: str,
    userid: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await db.execute(
            async_select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalars().first()
        if not conversation:
            raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다")
        if conversation.user_id != userid:
            raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
        history = []
        if conversation.conversation_log:
            try:
                history = json.loads(conversation.conversation_log)
            except json.JSONDecodeError:
                history = []
        return {
            "conversation_id": conversation.id,
            "image_title": conversation.image_title,
            "vlm_description": conversation.vlm_description,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "messages": history
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# ── 웹소켓 연결 엔드포인트 ──
@router.websocket("/ws/chat/{userid}")
async def websocket_endpoint(websocket: WebSocket, userid: str):
    await manager.connect(websocket, userid)
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                if message_data.get("message_type") == "ping":
                    await manager.send_message(userid, {"message_type": "pong"})
                    continue
                async with async_session() as db:
                    request = message_data.get("request", "")
                    image_title = message_data.get("image_title", "unknown")
                    vlm_description = message_data.get("vlm_description")
                    conversation_id = message_data.get("conversation_id")
                    
                    conversation = await get_or_create_conversation(
                        db, userid, image_title, vlm_description, conversation_id
                    )
                    conversation_history = await get_conversation_history(db, conversation.id)
                    await append_to_conversation_log(db, conversation, "user", request)
                    conversation_history.append({"role": "user", "content": request})
                    
                    user_requests = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
                    user_requests.append(request)
                    
                    reaction, question = generate_vts_response(request, user_requests)
                    response = f"{reaction}\n{question}"
                    print(response)
                    
                    await append_to_conversation_log(db, conversation, "assistant", response)
                    conversation_history.append({"role": "assistant", "content": response})
                    
                    conversation.updated_at = datetime.utcnow()
                    await db.commit()
                    
                    session_id = f"{userid}/{image_title}"
                    conversation_sessions[session_id] = {
                        "conversation_id": conversation.id,
                        "messages": conversation_history,
                        "last_activity": time.time()
                    }
                    await manager.send_message(userid, {
                        "message_type": "chat_response",
                        "conversation_id": conversation.id,
                        "session_id": session_id,
                        "response": response,
                    })
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
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-3:]])
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

async def cleanup_sessions():
    while True:
        try:
            current_time = time.time()
            expired_sessions = []
            for session_id, session_data in conversation_sessions.items():
                if current_time - session_data["last_activity"] > 1800:
                    expired_sessions.append(session_id)
            for session_id in expired_sessions:
                del conversation_sessions[session_id]
            await asyncio.sleep(300)
        except Exception as e:
            print(f"세션 정리 중 오류 발생: {str(e)}")
            await asyncio.sleep(300)

def init_app(app):
    @app.on_event("startup")
    async def startup_event():
        await init_database()
        asyncio.create_task(cleanup_sessions())
