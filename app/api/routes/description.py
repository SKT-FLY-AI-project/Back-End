from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel
from PIL import Image
import numpy as np
from app.config import UPLOAD_DIR
from app.utils.opencv_utils import load_and_preprocess_image, detect_edges, extract_dominant_colors, detect_painting_region
from app.utils.cnn_utils import predict_image
from app.utils.llm_utils import generate_vlm_description_qwen, generate_rich_description, text_to_speech
from app.utils.s3_utils import upload_to_s3

import shutil
import json
import time
import sys
import os
import cv2

# 분석 결과를 묶는 클래스 정의
class ImageAnalysisResult(BaseModel):
    vlm_description: str  # 이미지에 대한 설명
    dominant_colors: list  # 주요 색상
    edges: list  # 엣지 감지 결과

class ChatRequest(BaseModel):
    user_question: str  # 사용자가 입력한 질문
    
class MessageRequest(BaseModel):
    message: str

router = APIRouter(prefix="/chat", tags=["Chatbot"])

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

processing_status = {}  # 진행 상태 저장
@router.get("/status/{userid}")
async def get_status(userid: str):
    """
    특정 사용자의 진행 상태 반환
    """
    return {"status": processing_status.get(userid, "대기 중")}

@router.post("/describe/{userid}")
async def describe_image(userid: str, file: UploadFile = File(...)):
    """
    이미지를 업로드하면 분석을 수행하고 결과를 반환하는 API 엔드포인트.
    """
    # yield "START|이미지 업로드 중...\n"
    file_name = file.filename
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file.file.close()
    file = open(file_path, "rb")
    
    async def process_image():
        print(1)
        yield json.dumps({"status": "이미지 전처리 중...", "completed": False}) + "\n"
        image_pre = load_and_preprocess_image(file_path)
        image = detect_painting_region(image_pre)
        edges = detect_edges(image)
        dominant_colors = extract_dominant_colors(image)

        print(2)
        yield json.dumps({"status": "이미지 저장 및 업로드 중...", "completed": False}) + "\n"
        processed_file_path = f"{UPLOAD_DIR}/processed_{file_name}"
        cv2.imwrite(processed_file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        s3_url = upload_to_s3(userid, processed_file_path, f"{userid}/processed_{file_name}")

        print(3)
        yield json.dumps({"status": "설명 생성 중...", "completed": False}) + "\n"
        vlm_description = generate_vlm_description_qwen(file_path)
        if isinstance(vlm_description, str):
            vlm_description = [vlm_description]
        if not vlm_description:
            vlm_description = ["설명을 생성할 수 없습니다."]

        print(4)
        yield json.dumps({"status": "작품 제목 분석 중...", "completed": False}) + "\n"
        title = predict_image(image)
        if isinstance(title, set):
            title = list(title)[0]
        else:
            title = "제목 없음"

        print(5)
        yield json.dumps({"status": "풍부한 설명 생성 중...", "completed": False}) + "\n"
        rich_description = generate_rich_description(title, vlm_description[0], dominant_colors, edges)

        print(6)
        yield json.dumps({"status": "음성 변환 중...", "completed": False}) + "\n"
        audio_filename = f"output_{os.path.basename(file_path)}.mp3"
        audio_path = f"uploads/{audio_filename}"
        # text_to_speech(rich_description, output_file=audio_path)

        print(7)
        final_result = {
            "status": "완료",
            "completed": True,
            "data": {
                "image_url": s3_url,
                "title": title,
                "vlm_description": vlm_description[0],
                "rich_description": rich_description,
                "dominant_colors": dominant_colors.tolist(),
                # "edges_detected": "명확히 탐지됨" if edges.sum() > 10000 else "불명확",
                "audio_url": f"/static/{audio_filename}"
            }
        }
        print(8)
        yield json.dumps(final_result) + "\n"
    print(9)
    return StreamingResponse(process_image(), media_type="text/event-stream")

@router.post("/bot")
async def chatbot(request: Request):
    """
    사용자가 질문을 하면, 분석된 이미지 결과를 바탕으로 답변을 생성하는 API 엔드포인트.
    """
    body = await request.json()  # 요청 본문을 JSON으로 변환
    print("📥 받은 요청:", body)  # 요청 본문 출력

    try:
        request_data = MessageRequest(**body)  # BaseModel에 맞게 변환
    except Exception as e:
        print("❌ 요청 데이터 변환 오류:", e)
        return {"error": "잘못된 요청 형식입니다."}

    prompt = f"""
        사용자의 질문: "{request_data.message}"
        
        위 정보를 기반으로 사용자의 질문에 대해 상세하고 유익한 답변을 제공하세요.
        """
    
    answer = generate_rich_description("분석된 그림", prompt, [], [])
    
    print("\n💬 AI의 답변:")
    print(answer)

    return json.loads(json.dumps({"response": answer}, ensure_ascii=False))


@router.post("/user-prompt")
async def user_prompt(request: ChatRequest, analysis_result: ImageAnalysisResult):
    """
    사용자가 질문을 하면, 분석된 이미지 결과를 바탕으로 답변을 생성하는 API 엔드포인트.
    """
    
    # 전달된 분석 결과를 이용해 프롬프트 생성
    # prompt = f"""
    #     사용자는 '{analysis_result.vlm_description}' 작품에 대해 질문하고 있습니다.
    #     작품 설명: {analysis_result.vlm_description}
    #     주요 색상: {analysis_result.dominant_colors}
    #     엣지 감지 결과: {analysis_result.edges}
        
    #     사용자의 질문: "{request.user_question}"
        
    #     위 정보를 기반으로 사용자의 질문에 대해 상세하고 유익한 답변을 제공하세요.
    #     """
    prompt = f"""
        사용자는 '{analysis_result.vlm_description}' 작품에 대해 질문하고 있습니다.
        작품 설명: {analysis_result.vlm_description}
        주요 색상: {analysis_result.dominant_colors}
        
        사용자의 질문: "{request.user_question}"
        
        위 정보를 기반으로 사용자의 질문에 대해 상세하고 유익한 답변을 제공하세요.
        """
    
    # LLM을 이용한 답변 생성
    answer = generate_rich_description("분석된 그림", prompt, analysis_result.dominant_colors, analysis_result.edges)
    print("\n💬 AI의 답변:")
    print(answer)

    # 음성 변환
    audio_filename = f"answer_{os.path.basename(analysis_result.vlm_description)}.mp3"
    audio_path = f"uploads/{audio_filename}"
    text_to_speech(answer, output_file=audio_path)
    
    # 🔹 결과 JSON 반환
    return {
        "image_path": analysis_result.vlm_description,
        "vlm_description": analysis_result.vlm_description,
        "rich_description": answer,
        "dominant_colors": analysis_result.dominant_colors,
        # "edges_detected": "명확히 탐지됨" if sum(analysis_result.edges) > 10000 else "불명확",
        "audio_url": f"/static/{audio_filename}"  # 프론트엔드에서 음성 파일 접근 가능하도록 URL 제공
    }
