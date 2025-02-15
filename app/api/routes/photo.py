from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import io
import os
import boto3

# AWS S3 설정
AWS_ACCESS_KEY = "YOUR_AWS_ACCESS_KEY"
AWS_SECRET_KEY = "YOUR_AWS_SECRET_KEY"
S3_BUCKET_NAME = "YOUR_S3_BUCKET_NAME"
S3_REGION = "YOUR_AWS_REGION"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

router = APIRouter(prefix="/photo", tags=["Photo"])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tmp", "mnist_model.h5")

UPLOAD_DIR = "./app/static"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load the model once at the start
model = tf.keras.models.load_model(MODEL_PATH)

def upload_to_s3(userid: str, file_path: str, file_name: str):
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, f"{userid}/{file_name}")
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{userid}/{file_name}"
        return s3_url
    except Exception as e:
        print("S3 Upload Error:", e)
        return None

def preprocess_image(image: Image):
    image = image.convert("L")
    image = image.resize(((28, 28)))
    image_np = np.array(image)
    image_np = image_np.astype("float32") / 255.0
    image_np = np.expand_dims(image_np, axis=-1)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def detect_painting_region(image):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_regions = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            detected_regions.append(cv2.boundingRect(approx))
    
    if detected_regions:
        x, y, w, h = max(detected_regions, key=lambda r: r[2] * r[3])
        return Image.fromarray(image_np[y:y+h, x:x+w])
    else:
        return image

@router.post("/upload/")
async def upload(file: UploadFile = File(...)):
    if not file.content_type.startswith('image'):
        return {"error": "Invalid file type. Please upload an image."}
    
    contents = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    
    s3_url = upload_to_s3("user_id", file_path, file.filename)  # Pass user_id appropriately
    if s3_url:
        return {"message": "File uploaded successfully", "s3_url": s3_url}
    else:
        return {"error": "Failed to upload to S3"}

@router.post("/classify/{userid}")
async def classify(userid: str, file: UploadFile = File(...)):
    if not file.content_type.startswith('image'):
        return {"error": "Invalid file type. Please upload an image."}
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    painting = detect_painting_region(image)
    processed_image = preprocess_image(painting)
    
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    output_filename = f"detected_{file.filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    painting.save(output_path)
    
    s3_url = upload_to_s3(userid, output_path, output_filename)
    if s3_url:
        return {
            "status": 200,
            "data": {
                "predicted_class": int(predicted_class),
                "s3_url": s3_url
            },
            "message": "Success"
        }
    else:
        return {"error": "Failed to upload to S3"}
