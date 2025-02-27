import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO  # YOLOv8 사용

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 1. 모델 로드
model = YOLO("models/artwork_detector.pt")  # 미술 작품 감지를 위한 모델 경로 지정

# 2. 이미지 불러오기
image_path = "test_demo.jpg"  # 테스트할 이미지 경로
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. 추론 수행
results = model(image_path)

# 4. Bounding Box 그리기
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표 추출
        conf = box.conf[0]  # confidence score
        label = int(box.cls[0])  # 클래스 인덱스

        # 박스 그리기
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_rgb, f"Conf: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 5. 결과 출력
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
