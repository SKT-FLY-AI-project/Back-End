from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
import shutil
import json
import sys
import os
import cv2

from app.config import UPLOAD_DIR
from app.utils.image_processing import load_and_preprocess_image, detect_edges, extract_dominant_colors, detect_painting_region
from app.utils.artwork_search import search_artwork_by_title
from app.services.cnn_service import predict_image
from app.services.llm_service import generate_vlm_description_qwen, generate_rich_description
from app.managers.s3_manager import s3_manager

router = APIRouter(
    prefix="/api/describe", 
    tags=["Description"]
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@router.post("/{userid}/")
async def describe_image(userid: str, file: UploadFile = File(...)):
    """
    이미지를 업로드하면 분석을 수행하고 결과를 반환하는 API 엔드포인트.
    """
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
        s3_url = s3_manager.upload_file(processed_file_path, f"{userid}/processed_{file_name}")

        yield json.dumps({"status": "설명 생성 중...", "completed": False}) + "\n"
        vlm_description = await generate_vlm_description_qwen(file_path)
        if isinstance(vlm_description, str):
            vlm_description = [vlm_description]
        if not vlm_description:
            vlm_description = ["설명을 생성할 수 없습니다."]

        yield json.dumps({"status": "작품 제목 분석 중...", "completed": False}) + "\n"
        title = await predict_image(image)
        artist = ""
        period = ""
        webpage = ""
        artwork_info = {}
        if isinstance(title, set):
            title = list(title)[0]
            artwork_info = search_artwork_by_title(title)
        # else: 
        #     
        
        
        if title == "스키아 보니의 해안": artist = "미셸 앙리"
        elif title == "여름의 베퇴유": artist = "미셸 앙리"
        elif title == "여름이 다가옵니다": artist = "미셸 앙리"
        elif title == "여름이 다가옵니다": artist = "미셸 앙리"
        elif title == "장미와 유리병들": artist = "미셸 앙리"
        elif title == "캐나다의 사과": artist = "미셸 앙리"
        elif title == "파리 인 블루": artist = "미셸 앙리"
        elif title == "이즈미르에서의 정박": artist = "미셸 앙리"
        else:
            title = artwork_info.get("title") if artwork_info else None
            artist = artwork_info.get("artist") if artwork_info else None
            period = artwork_info.get("period") if artwork_info else None
            webpage = artwork_info.get("webpage") if artwork_info else None
        
        yield json.dumps({"status": "풍부한 설명 생성 중...", "completed": False}) + "\n"
        rich_description = await generate_rich_description(title, artist, period, webpage, vlm_description[0], dominant_colors, edges)

        final_result = {
            "status": "완료",
            "completed": True,
            "data": {
                "image_url": s3_url,
                "title": title,
                "artist": artist,
                "vlm_description": vlm_description[0],
                "rich_description": rich_description,
                "dominant_colors": dominant_colors.tolist()
            }
        }
        yield json.dumps(final_result) + "\n"
    
    return StreamingResponse(process_image(), media_type="text/event-stream")