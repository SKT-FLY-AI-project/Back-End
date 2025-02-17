from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from app.config import UPLOAD_DIR
from app.utils.s3_utils import upload_to_s3
from app.utils.model_utils import model_manager
from app.utils.image_utils import detect_painting_region, preprocess_image

import io
import os

router = APIRouter(prefix="/photo", tags=["Photo"])

@router.post("/upload/{userid}")
async def upload(userid: str, file: UploadFile = File(...)):
    if not file.content_type.startswith("image"):
        return {"error": "Invalid file type. Please upload an image."}
    
    contents = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(contents)
    
    s3_url = upload_to_s3(userid, file_path, file.filename)
    if s3_url:
        return {"message": "File uploaded successfully", "s3_url": s3_url}
    return {"error": "Failed to upload to S3"}

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

@router.post("/classify/{userid}")
async def classify(userid: str, file: UploadFile = File(...)):
    if not file.content_type.startswith("image"):
        return {"error": "Invalid file type. Please upload an image."}
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    # painting = detect_painting_region(image)
    
    processed_image = preprocess_image(image)
    model = model_manager.load_model()
    prediction = model.predict(processed_image)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    
    output_filename = f"{userid}/detected{predicted_class}_{file.filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    image.save(output_path)
    
    s3_url = upload_to_s3(userid, output_path, output_filename)
    if s3_url:
        print(s3_url)
        return {
            "status": 200,
            "data": {
                "predicted_class": predicted_class,
                "image_url": s3_url
            },
            "message": "Success"
        }
    return {"error": "Failed to upload to S3"}