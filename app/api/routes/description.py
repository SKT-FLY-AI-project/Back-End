from fastapi import APIRouter, UploadFile, File, Depends
from PIL import Image
import numpy as np
from app.config import UPLOAD_DIR
from app.utils.opencv_utils import load_and_preprocess_image, detect_edges, extract_dominant_colors
from app.utils.llm_utils import generate_vlm_description_qwen, generate_rich_description, text_to_speech

import shutil
import sys
import os

router = APIRouter(prefix="/desc", tags=["Description"])

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def process_image(image_path):
    """
    이미지를 받아서 분석 후 결과를 JSON 형태로 반환하는 함수.
    """
    # 🔹 이미지 전처리 및 OpenCV 분석
    image = load_and_preprocess_image(image_path)
    edges = detect_edges(image)
    dominant_colors = extract_dominant_colors(image)

    # ✅ Qwen2.5-VL 실행하여 설명 생성
    vlm_description = generate_vlm_description_qwen(image_path)

    if isinstance(vlm_description, str):
        vlm_description = [vlm_description]

    # ✅ 설명이 없을 경우 기본값 설정
    if not vlm_description:
        vlm_description = ["설명을 생성할 수 없습니다."]

    # 🔹 LLM을 활용한 설명 생성
    rich_description = generate_rich_description("분석된 그림", vlm_description[0], dominant_colors, edges)

    # 🔹 음성 변환 실행 (음성 파일 저장)
    audio_filename = f"output_{os.path.basename(image_path)}.mp3"
    audio_path = f"uploads/{audio_filename}"
    text_to_speech(rich_description, output_file=audio_path)

    # 🔹 결과 JSON 반환
    return {
        "image_path": image_path,
        "vlm_description": vlm_description[0],
        "rich_description": rich_description,
        "dominant_colors": dominant_colors.tolist(),
        "edges_detected": "명확히 탐지됨" if edges.sum() > 10000 else "불명확",
        "audio_url": f"/static/{audio_filename}"  # 프론트엔드에서 음성 파일 접근 가능하도록 URL 제공
    }


@router.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    """
    이미지를 업로드하면 분석을 수행하고 결과를 반환하는 API 엔드포인트.
    """
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    
    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔹 이미지 분석 실행
    print(file_path)
    result = process_image(file_path)

    return result  # JSON 응답 반환