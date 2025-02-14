from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import io
import os

router = APIRouter(prefix="/photo", tags=["Photo"])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tmp", "mnist_model.h5")

UPLOAD_DIR = "./app/static"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load the model once at the start
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image):
    image = image.convert("L")  # Convert image to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_np = np.array(image)
    image_np = image_np.astype("float32") / 255.0
    image_np = np.expand_dims(image_np, axis=-1)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def detect_painting_region(image: Image.Image, min_area_ratio=0.2):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_regions = [
        cv2.boundingRect(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) 
        for cnt in contours 
        if len(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) == 4
    ]
    
    if detected_regions:
        # 가장 큰 사각형 영역 선택
        x, y, w, h = max(detected_regions, key=lambda r: r[2] * r[3])
        
        # 전체 이미지 대비 크롭된 영역이 너무 작으면 원본 반환
        img_area = image_np.shape[0] * image_np.shape[1]
        cropped_area = w * h
        if cropped_area < min_area_ratio * img_area:
            print("⚠️ 그림 영역이 너무 작아 원본 이미지를 반환합니다.")
            return image
        return Image.fromarray(image_np[y:y+h, x:x+w])
    return image

@router.post("/upload/")
async def upload(file: UploadFile = File(...)):
    print(file.content_type)
    if not file.content_type.startswith('image'):
        return {"error": "Invalid file type. Please upload an image."}
    
    contents = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    
    return {"message": f"File uploaded successfully: {file.filename}"}

@router.post("/detect/")
async def detect(file: UploadFile = File(...)):
    print("Detect:", file.content_type)
    if not file.content_type.startswith('image'):
        return {"error": "Invalid file type. Please upload an image."}
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    painting = detect_painting_region(image)
    
    img_byte_arr = io.BytesIO()
    painting.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")

@router.post("/classify/")
async def classify(file: UploadFile = File(...)):
    print("Hello,", file.filename, file.content_type)
    if not file.content_type.startswith('image'):
        return {
            "status": 400,
            "error": "Invalid file type. Please upload an image."
        }
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    painting = detect_painting_region(image)
    
    # 입력 이미지 전처리
    processed_image = preprocess_image(painting)
    print(f"Processed image shape: {processed_image.shape}")
    
    # 모델 예측
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    print("Predicted class:", predicted_class)

    # 이미지를 로컬에 저장
    output_filename = f"detected_{file.filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    painting.save(output_path)
    print("Saved as", output_path)

    # 이미지를 스트리밍할 URL 반환
    image_url = f"http://YOUR_SERVER_URI/static/{output_filename}"

    return {
        "status": 200,
        "data": {
            "predicted_class": int(predicted_class),
            "image_url": image_url  # 클라이언트는 이 URL을 통해 이미지를 접근
        },
        "message": "Success"
    }
