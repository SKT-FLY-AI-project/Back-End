from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
import json

from app.database import get_db
from app.managers.websocket_manager import manager
from app.crud.chat_crud import get_or_create_conversation, save_message, get_conversation_history, get_user_conversations
from app.services.llm_service import generate_vts_response, classify_user_input, answer_user_question, load_precomputed_data

router = APIRouter(
    prefix='/ws/chat',
    tags=["WebSocket"]    
)

# 웹소켓 연결 엔드포인트
@router.websocket("/{userid}")
async def websocket_endpoint(
    websocket: WebSocket, 
    userid: str,
    image_url: str = None,
    title: str = None,
    artist: str = None,
    rich_description: str = None,
    conversation_id: str = None
):
    # print(1)
    await manager.connect(websocket, userid)
    vector_store = await load_precomputed_data()
    try:
        db_gen = get_db()                   # 제너레이터 생성
        db = await db_gen.__anext__()
        while True:
            try:
                # print(2)
                # 클라이언트로부터 메시지 수신
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # print(3)
                # print(message_data.get("message_type"))
                # 메시지 타입에 따른 처리
                if message_data.get("message_type") == "ping":
                    await manager.send_message(userid, {"message_type": "pong"})
                    continue
                
                print(4)
                print(message_data)
                
                request = message_data.get("request", "")
                print(111)
                image_url = message_data.get("image_url")
                print(222)
                title = message_data.get("title", "제목 없음")
                print(300)
                artist = message_data.get("artist", "작자 미상")
                print(333)
                rich_description = message_data.get("rich_description")
                print(444)
                conversation_id = message_data.get("conversation_id")
                
                print(5)
                # 대화 얻기 또는 생성
                conversation = await get_or_create_conversation(
                    db, userid, image_url, title, artist, rich_description, conversation_id
                )
                
                print(6)
                # 기존 대화 기록 불러오기
                conversation_history = await get_conversation_history(db, conversation.id)
                
                print(7)
                # 새로운 메시지 추가
                await save_message(db, conversation.id, "user", request)
                user_requests = [msg["content"] for msg in conversation_history]
                user_requests.append(request)
                
                print(8)
                # LLM으로 응답 생성
                input_type = classify_user_input(request)
                print(input_type)
                if input_type in ("info", "mixed"):
                    response = answer_user_question(request, user_requests, title, artist, rich_description, vector_store)
                
                elif input_type in ("feeling", "unknown"):
                    reaction, next_question = generate_vts_response(request, user_requests)
                    response = reaction + "\n" + next_question

                print(9)
                print(input_type)
                print(response)
                # 응답을 대화 기록에 저장
                await save_message(db, conversation.id, "assistant", response)
                
                print(10)
                # 대화 최종 업데이트 시간 갱신
                conversation.updated_at = datetime.utcnow()
                await db.commit()
                
                print(11)
                # 응답 전송
                await manager.send_message(userid, {
                    "message_type": "chat_response",
                    "conversation_id": conversation.id,
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
    finally:
            await db_gen.aclose()