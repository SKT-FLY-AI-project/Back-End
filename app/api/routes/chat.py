from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from groq import Groq
from app.models.llm_model import LLMModel
from app.utils.llm_utils import generate_vlm_description_qwen, generate_rich_description
import os
import json
import time

router = APIRouter(prefix="/chat", tags=["Chat"])
os.environ['HF_HOME'] = "D:/huggingface_models"
client = Groq(api_key="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")

class ChatMessage(BaseModel):
    """VTS 대화 세션 요청 데이터 모델"""
    request: str
    # conversation_history: Optional[List[Dict[str, str]]] = None
    image_title: str
    vlm_description: str
    dominant_colors: List

class SessionRequest(BaseModel):
    """대화 세션 관리 요청 모델"""
    session_id: str


# 세션 만료 시간 설정 (초)
SESSION_TIMEOUT = 3600  # 1시간

# 대화 세션 저장 딕셔너리
conversation_sessions = {}


def clear_expired_sessions():
    """오래된 세션을 정리하는 함수"""
    current_time = time.time()
    expired_sessions = [
        session_id
        for session_id, data in conversation_sessions.items()
        if current_time - data["last_activity"] > SESSION_TIMEOUT
    ]

    for session_id in expired_sessions:
        del conversation_sessions[session_id]
        print(f"🗑️ 오래된 세션 {session_id} 삭제됨")


@router.post("/create_session/{userid}/{photo_id}")
async def create_session(userid: str, photo_id: str):
    """새로운 대화 세션을 생성하는 엔드포인트"""
    session_id = f"{userid}_{photo_id}"

    # 기존 세션 확인
    if session_id in conversation_sessions:
        return {
            "session_id": session_id,
            "message": "이미 존재하는 세션이 있습니다.",
            "conversation": conversation_sessions[session_id]["messages"],
        }

    # 새로운 세션 생성
    conversation_sessions[session_id] = {"messages": [], "last_activity": time.time()}

    return {"session_id": session_id, "message": "새로운 세션이 생성되었습니다."}


@router.get("/check_session/{userid}/{photo_id}")
async def check_session(userid: str, photo_id: str):
    """세션 존재 여부를 확인하는 엔드포인트"""
    session_id = f"{userid}_{photo_id}"
    if session_id in conversation_sessions:
        return {
            "exists": True,
            "session_id": session_id,
            "conversation": conversation_sessions[session_id]["messages"],
        }
    return {"exists": False, "message": "세션이 존재하지 않습니다."}


@router.post("/end_session")
async def end_session(request: SessionRequest):
    """대화 세션을 종료하는 엔드포인트"""
    session_id = request.session_id
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        return {"message": f"세션 {session_id}이 종료되었습니다."}

    raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")


def load_vts_questions():
    # 현재 파일(llm.py)이 있는 폴더 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))  # three_llm 폴더 경로

    # JSON 파일 경로 설정
    file_path = os.path.join(current_dir, "data", "VTS_RAG_questions.json")

    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ RAG 질문 파일을 찾을 수 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# 1-2. 미술 정보 RAG에서 질문을 가져오는 함수 (예시) : art_RAG_questions.json
def load_art_questrion(query):
    """
    사용자의 감상 및 질문에 대해 관련 미술 정보를 검색하는 함수.
    """
    # 예시임. # art_RAG_question.json 파일 가져오는 코드로 수정하자.
    # retriever = FAISS.load_local("art_database", embeddings).as_retriever()
    # results = retriever.get_relevant_documents(query)
    # return results[0].page_content if results else "관련된 미술 정보를 찾지 못했습니다."


# 2. 질문 유형 파악하여 -> 알맞은 질문 형성하기
def retrieve_question(
    user_requests, image_title, vlm_description, dominant_colors, edges=[]
):
    """
    사용자의 이전 답변을 기반으로 적절한 'VTS 질문'을 RAG에서 검색
    """
    """
    - 사용자의 입력을 분석하여 VTS 질문을 생성할지, 미술 정보를 제공할지 결정.
    - 감정적 공감 또는 정보 제공이 필요한 경우: 미술 정보 검색
    - 질문을 생성할 경우: VTS 질문 매뉴얼 검색
    """
    # 사용자의 마지막 질문 (AI가 질문 생성을 위해 넣어줘야할 값)
    print(user_requests)
    previous_responses = user_requests[-1]
    
    rag_questions = (
        load_vts_questions()
    )  # 일단 테스트용으로 여기다 두긴 하는데... 나중에 정리합시다.
    # RAG에서 검색 (현재는 임시로 JSON에서 질문을 선택하는 형태)

    # 🎨 2-1. 사용자가 작품 설명을 요구함.
    if (
        "느낌" in previous_responses
        or "설명" in previous_responses
        or "배경" in previous_responses
    ):  # NLP로 개선하기
        relevant_questions = []
        # 👉 RAG : 작품 정보 설명하기.
        relevant_questions = load_art_questrion(previous_responses)
        # 👉 RAG : VTS의 질문을 전달.
        relevant_questions = [q for q in rag_questions if q["classification"] == "질문"]

        print(f"📚 ART AI LLM : {relevant_questions}")
        return relevant_questions

    # 🎨 2-2. 사용자가 자신의 생각을 말함.
    else:
        relevant_questions = []
        # 👉 RAG : : VTS의 반응와 관련된 말을 전달.
        # 사용자의 이전 답변을 분석 (여기서는 단순히 랜덤으로 선택, 실제 구현 시 NLP 활용 가능)
        relevant_questions = [q for q in rag_questions if q["classification"] == "반응"]

        # 👉 LLM : AI가 작품에 대해 생각하는 말을 전다. + LLM기반 작품 정보
        prompt = f"""
        사용자와 '{image_title}' 작품에 대해 대화하고 있습니다.
        작품 설명: {vlm_description}
        주요 색상: {dominant_colors}
        엣지 감지 결과: {edges}
        
        사용자의 생각: "{previous_responses}"
        
        위 정보를 기반으로 사용자의 생각에 대해 유익한 답변을 제공하세요.
        사용자 생각에대해 동의 및 다른 의견을 제시해주세요.
        """

        # LLM을 이용한 답변 생성
        # answer = generate_rich_description(image_title, prompt, dominant_colors, edges)
        answer = generate_rich_description(image_title, prompt, dominant_colors)
        print("\n💬 AI의 답변:")
        print(answer)

        # 👉 RAG : VTS의 질문을 전달.
        # 첫 질문은 "전체에 대한 적극적인 관계 만들기"에서 선택
        relevant_questions = [q for q in rag_questions if q["classification"] == "질문"]
        return relevant_questions


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


@router.post("/bot/{userid}")
async def start_vts_conversation(userid: str, chat_data: ChatMessage = Body(...)):
    """VTS 방식의 감상 대화를 진행하는 엔드포인트"""
    try:
        request = chat_data.request
        session_id = f"{userid}/{chat_data.image_title}"

        # 기존 세션이 있으면 불러오기, 없으면 새로 생성
        if session_id in conversation_sessions:
            session_data = conversation_sessions[session_id]
        else:
            session_data = {"messages": [], "last_activity": time.time()}
            conversation_sessions[session_id] = session_data

        # 기존 대화 기록 불러오기 (세션이 있으면)
        conversation_history = session_data["messages"]
        
        # 새로운 메시지 추가
        conversation_history.append({"role": "user", "content": request})
        user_requests = [msg["content"] for msg in conversation_history]  # 메시지에서 content만 추출

        # LLM을 사용한 응답 생성
        reaction, question = generate_vts_response(request, user_requests)
        response = reaction + '\n' + question

        # 응답을 대화 기록에 저장
        conversation_history.append({"role": "assistant", "content": response})
        session_data["last_activity"] = time.time()  # 세션 최신화

        return {
            "session_id": session_id,
            "response": response,
            "conversation": conversation_history,  # 기존 기록 포함
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")