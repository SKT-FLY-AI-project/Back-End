# from fastapi import FastAPI, HTTPException, Body
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Optional
# import anthropic
# import os
# from dotenv import load_dotenv

# # .env 파일에서 환경 변수 로드
# load_dotenv()

# # Anthropic API 키 가져오기
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# if not ANTHROPIC_API_KEY:
#     raise ValueError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")

# # Anthropic 클라이언트 초기화
# client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# app = FastAPI()

# # CORS 설정
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 개발 환경에서는 모든 오리진 허용, 프로덕션에서는 특정 도메인으로 제한하세요
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatMessage(BaseModel):
#     message: str
#     conversation_history: Optional[List[Dict[str, str]]] = None

# # 대화 세션을 저장할 딕셔너리
# conversation_sessions = {}

# @app.post("/chat/bot")
# async def chat_with_bot(chat_data: ChatMessage = Body(...)):
#     try:
#         message = chat_data.message
        
#         # 클라이언트에서 대화 기록을 전송한 경우 사용
#         conversation_history = chat_data.conversation_history or []
        
#         # Anthropic 메시지 포맷으로 변환
#         messages = []
        
#         # 대화 기록이 있으면 메시지에 추가
#         for item in conversation_history:
#             if "question" in item and item["question"]:
#                 messages.append({"role": "user", "content": item["question"]})
#             if "response" in item and item["response"]:
#                 messages.append({"role": "assistant", "content": item["response"]})
        
#         # 현재 메시지 추가
#         messages.append({"role": "user", "content": message})
        
#         # 대화 기록이 비어있으면 시스템 메시지 추가
#         if not messages:
#             messages.insert(0, {
#                 "role": "system",
#                 "content": "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 한국어로 대답해주세요."
#             })
        
#         # Claude API 호출
#         response = client.messages.create(
#             model="claude-3-5-sonnet-20241022",  # 원하는 모델로 변경 가능
#             max_tokens=1000,
#             messages=messages
#         )
        
#         # 응답 반환
#         return {"response": response.content[0].text}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# # 대화 세션 관리 엔드포인트 (선택사항)
# class SessionRequest(BaseModel):
#     session_id: str

# @app.post("/chat/create_session")
# async def create_session(request: SessionRequest):
#     session_id = request.session_id
#     if session_id not in conversation_sessions:
#         conversation_sessions[session_id] = []
#     return {"session_id": session_id, "message": "세션이 생성되었습니다."}

# @app.post("/chat/end_session")
# async def end_session(request: SessionRequest):
#     session_id = request.session_id
#     if session_id in conversation_sessions:
#         del conversation_sessions[session_id]
#     return {"message": "세션이 종료되었습니다."}

# # 서버 실행용 코드
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
# # def answer_user_question(image_title, vlm_description, dominant_colors, edges):
# #     """사용자의 질문을 받아 LLM을 통해 답변을 생성하는 함수"""
# #     while True:
# #         user_question = input("\n❓ 추가 질문 (종료하려면 'exit' 입력): ")
# #         if user_question.lower() == "exit":
# #             print("📢 질문 모드 종료.")
# #             break
        
# #         # LLM에 전달할 프롬프트 생성
# #         prompt = f"""
# #         사용자는 '{image_title}' 작품에 대해 질문하고 있습니다.
# #         작품 설명: {vlm_description}
# #         주요 색상: {dominant_colors}
# #         엣지 감지 결과: {edges}
        
# #         사용자의 질문: "{user_question}"
        
# #         위 정보를 기반으로 사용자의 질문에 대해 상세하고 유익한 답변을 제공하세요.
# #         """
        
# #         # LLM을 이용한 답변 생성
# #         answer = generate_rich_description(image_title, prompt, dominant_colors, edges)
# #         print(answer)

# #         # 음성 변환
# #         text_to_speech(answer, output_file=f"answer_{image_title}.mp3")