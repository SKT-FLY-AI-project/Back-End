
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
from typing import Optional
from PIL import Image
from ultralytics import YOLO
import os

router = APIRouter(
    prefix="/api/camera",
    tags=["Auth"]
)

# 모델 로드
model = YOLO("models/artwork_detector.pt")  # 미술 작품 감지를 위한 모델 경로

@router.post("/detect/")
async def detect(
    file: UploadFile = File(...),
    max_objects: int = Query(1, description="최대 감지할 객체 수")
):
    """
    이미지에서 객체를 감지하고 가장 신뢰도가 높은 객체 하나만 반환합니다.
    객체가 중앙에 위치해 있는지 여부와 중앙으로 이동하기 위한 방향 안내를 제공합니다.
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = np.array(Image.open(io.BytesIO(contents)))
        
        # BGR로 변환 (YOLOv8이 RGB를 기대하므로 필요한 경우에만)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
            
        # 이미지 크기 가져오기
        height, width = image.shape[:2]
        
        # 중앙점 계산
        center_x = width / 2
        center_y = height / 2
        
        # 모델 추론
        results = model(image)
        
        # 감지 결과가 없을 경우
        if len(results) == 0 or len(results[0].boxes) == 0:
            return {"detected": False, "message": "객체를 찾을 수 없습니다."}
        
        # 감지된 객체 중 가장 신뢰도가 높은 객체 찾기
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        
        # 신뢰도에 따라 정렬하고 가장 높은 것 선택
        if max_objects > len(confidences):
            max_objects = len(confidences)
            
        # 가장 신뢰도가 높은 객체 하나만 선택
        best_idx = np.argmax(confidences)
        
        # 바운딩 박스 정보 추출
        box = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        # 감지된 객체의 중심점
        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2
        
        # 중앙으로부터의 거리 계산 (상대적 거리)
        distance_x = (object_center_x - center_x) / width
        distance_y = (object_center_y - center_y) / height
        
        # 객체가 중앙에 있는지 확인 (중앙에서 ±5% 이내)
        is_centered = abs(distance_x) < 0.05 and abs(distance_y) < 0.05
        
        # 중앙으로 이동하기 위한 방향 안내
        direction = "객체를 "
        
        if abs(distance_x) >= 0.05:
            if distance_x < 0:
                direction += "오른쪽"
            else:
                direction += "왼쪽"
                
            if abs(distance_y) >= 0.05:
                direction += "과 "
                
        if abs(distance_y) >= 0.05:
            if distance_y < 0:
                direction += "아래쪽"
            else:
                direction += "위쪽"
                
        direction += "으로 이동하세요"
        
        if is_centered:
            direction = "객체가 중앙에 위치했습니다."
        
        # 객체 클래스와 신뢰도
        confidence = float(confidences[best_idx])
        class_id = int(boxes.cls[best_idx].item())
        
        # 응답 데이터
        response_data = {
            "detected": True,
            "centered": is_centered,
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "center_x": int(object_center_x),
            "center_y": int(object_center_y),
            "frame_width": width,
            "frame_height": height,
            "confidence": confidence,
            "class_id": class_id,
            "direction": direction
        }
        
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"객체 감지 중 오류 발생: {str(e)}")
