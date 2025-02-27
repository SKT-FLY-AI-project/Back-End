from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
from typing import Optional
from PIL import Image
from ultralytics import YOLO
import os
from app.config import YOLO_PATH
import matplotlib.pyplot as plt
import uuid

router = APIRouter(
    prefix="/api/camera",
    tags=["Camera"]  # Auth에서 Camera로 태그 수정
)

# 모델 로드
model = YOLO(YOLO_PATH)  # 미술 작품 감지를 위한 모델 경로

@router.post("/detect/")
async def detect(
    file: UploadFile = File(...),
    max_objects: int = Query(1, description="최대 감지할 객체 수")
):
    """
    이미지에서 객체를 감지하고 가장 큰 객체 하나만 반환합니다.
    객체가 중앙에 위치해 있는지 여부와 중앙으로 이동하기 위한 방향 안내를 제공합니다.
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = np.array(Image.open(io.BytesIO(contents)))
        
        # 이미지 크기 가져오기
        height, width = image.shape[:2]
        
        # 중앙점 계산
        center_x = width / 2
        center_y = height / 2
        
        # 모델 추론
        results = model(image)
        
        # 시각화를 위한 이미지 복사본 만들기
        visualized_image = image.copy()
        
        # 감지 결과가 없을 경우
        if len(results) == 0 or len(results[0].boxes) == 0:
            return {"detected": False, "message": "객체를 찾을 수 없습니다."}
        
        # 감지된 객체들의 정보 추출
        boxes = results[0].boxes
        all_boxes = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        
        # 각 바운딩 박스의 면적 계산
        areas = []
        for i, box in enumerate(all_boxes):
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        
        # 가장 큰 객체의 인덱스 찾기
        best_idx = np.argmax(areas)
        
        # 바운딩 박스 정보 추출
        box = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        # 감지된 객체의 중심점
        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2
        
        # 중앙으로부터의 거리 계산 (상대적 거리)
        distance_x = (object_center_x - center_x) / width
        distance_y = (object_center_y - center_y) / height
        
        threshold = 0.1
        # 객체가 중앙에 있는지 확인 (중앙에서 ±10% 이내)
        is_centered = abs(distance_x) < threshold and abs(distance_y) < threshold
        
        # 객체의 크기가 적절한지 확인 (이미지 프레임 대비)
        area = (x2 - x1) * (y2 - y1)
        frame_area = width * height
        area_ratio = area / frame_area
        
        # 적정 거리 기준 (프레임 대비 객체 크기 비율)
        too_close = area_ratio > 0.9  # 전체 이미지의 70% 이상을 차지하면 너무 가까움
        too_far = area_ratio < 0.45    # 전체 이미지의 15% 미만을 차지하면 너무 멈
        
        # 중앙으로 이동하기 위한 방향 안내
        direction = "카메라를 "
        
        if abs(distance_x) >= threshold:
            if distance_x < 0:
                direction += "위쪽"
            else:
                direction += "아래쪽"
                
            if abs(distance_y) >= threshold:
                direction += "과 "
                
        if abs(distance_y) >= threshold:
            if distance_y < 0:
                direction += "오른쪽"
            else:
                direction += "왼쪽"
                
        direction += "으로 이동하세요"
        
        if is_centered:
            print(area, frame_area, area_ratio)
            print(too_far, too_close, type(area_ratio))
            if too_far:
                direction = "피사체가 너무 멉니다. 더 가까이 가세요."
            elif too_close:
                direction = "피사체가 너무 가깝습니다. 뒤로 이동하세요."
            else:
                direction = "피사체가 중앙에 위치했습니다."
        # else:
        #     if too_close:
        #         direction += ". 피사체가 너무 가깝습니다."
        #     elif too_far:
        #         direction += ". 피사체가 너무 멉니다."
        
        # 객체 클래스와 신뢰도
        confidence = float(confidences[best_idx])
        class_id = int(boxes.cls[best_idx].item())
        
        # 가장 큰 객체만 시각화 (디버깅용)
        try:
            # BGR로 변환 (OpenCV 시각화를 위해)
            if len(visualized_image.shape) == 3 and visualized_image.shape[2] == 3:
                visualized_image = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR)
            
            # 박스 그리기
            cv2.rectangle(visualized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(visualized_image, f"Conf: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 중앙 표시
            cv2.circle(visualized_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            # 객체 중심 표시
            cv2.circle(visualized_image, (int(object_center_x), int(object_center_y)), 5, (255, 0, 0), -1)
            
            # 이미지 저장 (디버깅 목적)
            # 고유한 파일 이름 생성
            filename = f"detection_{uuid.uuid4()}.jpg"
            static_dir = './static'
            
            # static 디렉토리가 존재하는지 확인하고 없으면 생성
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
                
            # 이미지 저장
            cv2.imwrite(os.path.join(static_dir, filename), visualized_image)
        except Exception as vis_error:
            print(f"시각화 중 오류 발생 (무시됨): {str(vis_error)}")
        
        # 응답 데이터
        response_data = {
            "detected": True,
            "centered": is_centered,
            "too_close": too_close,
            "too_far": too_far,
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "center_x": int(object_center_x),
            "center_y": int(object_center_y),
            "frame_width": width,
            "frame_height": height,
            "area_ratio": float(area_ratio),
            "confidence": confidence,
            "class_id": class_id,
            "direction": direction
        }
        
        return response_data
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"객체 감지 중 오류 발생: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"객체 감지 중 오류 발생: {str(e)}")