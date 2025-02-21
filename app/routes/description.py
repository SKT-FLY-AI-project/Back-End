from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from app.config import UPLOAD_DIR
from app.utils.opencv_utils import load_and_preprocess_image, detect_edges, extract_dominant_colors, detect_painting_region
from app.utils.cnn_utils import predict_image
from app.utils.llm_utils import generate_vlm_description_qwen, generate_rich_description
from app.utils.s3_utils import upload_to_s3

import shutil
import json
import sys
import os
import cv2

router = APIRouter(
    prefix="/api/describe", 
    tags=["Description"]
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@router.post("/{userid}")
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
        yield json.dumps({"status": "이미지 전처리 중...", "completed": False}) + "\n"
        image_pre = load_and_preprocess_image(file_path)
        image = detect_painting_region(image_pre)
        edges = detect_edges(image)
        dominant_colors = extract_dominant_colors(image)

        yield json.dumps({"status": "이미지 저장 및 업로드 중...", "completed": False}) + "\n"
        processed_file_path = f"{UPLOAD_DIR}/processed_{file_name}"
        cv2.imwrite(processed_file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        s3_url = upload_to_s3(userid, processed_file_path, f"{userid}/processed_{file_name}")

        yield json.dumps({"status": "설명 생성 중...", "completed": False}) + "\n"
        vlm_description = generate_vlm_description_qwen(file_path)
        if isinstance(vlm_description, str):
            vlm_description = [vlm_description]
        if not vlm_description:
            vlm_description = ["설명을 생성할 수 없습니다."]

        yield json.dumps({"status": "작품 제목 분석 중...", "completed": False}) + "\n"
        title = predict_image(image)
        if isinstance(title, set):
            title = list(title)[0]
        else:
            title = "제목 없음"

        yield json.dumps({"status": "풍부한 설명 생성 중...", "completed": False}) + "\n"
        rich_description = generate_rich_description(title, vlm_description[0], dominant_colors, edges)

        yield json.dumps({"status": "음성 변환 중...", "completed": False}) + "\n"
        audio_filename = f"output_{os.path.basename(file_path)}.mp3"
        # audio_path = f"uploads/{audio_filename}"
        # text_to_speech(rich_description, output_file=audio_path)

        final_result = {
            "status": "완료",
            "completed": True,
            "data": {
                "image_url": s3_url,
                "title": title,
                "vlm_description": vlm_description[0],
                "rich_description": rich_description,
                "dominant_colors": dominant_colors.tolist(),
                "audio_url": f"/static/{audio_filename}"
            }
        }
        yield json.dumps(final_result) + "\n"
    
    return StreamingResponse(process_image(), media_type="text/event-stream")