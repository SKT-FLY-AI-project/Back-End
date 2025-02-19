from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse
import io
from PIL import Image
import numpy as np
from app.config import UPLOAD_DIR
from app.utils.s3_utils import upload_to_s3
# from app.utils.image_utils import preprocess_image
# from app.utils.opencv_utils import load_and_preprocess_image, detect_painting_region

import base64
import io
import os

router = APIRouter(prefix="/photo", tags=["Photo"])

# @router.post("/upload/{userid}")
# async def upload(userid: str, file: UploadFile = File(...)):
#     if not file.content_type.startswith("image"):
#         return {"error": "Invalid file type. Please upload an image."}
    
#     contents = await file.read()
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
    
#     with open(file_path, "wb") as f:
#         f.write(contents)
    
#     s3_url = upload_to_s3(userid, file_path, file.filename)
#     if s3_url:
#         return {"message": "File uploaded successfully", "s3_url": s3_url} # 이걸 받아서 DB에서 저장해야함. 
#     return {"error": "Failed to upload to S3"}
\

# @router.post("/detect")
# async def detect(file: UploadFile = File(...)):
#     print("Detect:", file.content_type)
    
#     if not file.content_type.startswith('image'):
#         return JSONResponse(content={"error": "Invalid file type. Please upload an image."}, status_code=400)
    
#     contents = await file.read()
#     image_open = Image.open(io.BytesIO(contents)).convert("RGB")
#     image_np = np.array(image_open)
#     # image_pre = load_and_preprocess_image(image_np)
#     detect_painting = detect_painting_region(image_np) # imige_utils.py가 아닌, opencv_utils.py의 detect_painting_region() 함수로 대체
    
#     # numpy 배열을 PIL 이미지로 변환
#     painting = Image.fromarray(detect_painting)
#     img_byte_arr = io.BytesIO()
#     painting.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)

#     # ✅ Base64로 인코딩하여 반환
#     encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    
#     return JSONResponse(content={"image": encoded_image})


# @router.post("/classify/{userid}")
# async def classify(userid: str, file: UploadFile = File(...)):
#     if not file.content_type.startswith("image"):
#         return {"error": "Invalid file type. Please upload an image."}
    
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents))
#     painting = detect_painting_region(image)
    
#     processed_image = preprocess_image(painting)
#     model = model_manager.load_model()
#     prediction = model.predict(processed_image)
#     predicted_class = int(np.argmax(prediction, axis=1)[0])
    
#     output_filename = f"{userid}/detected{predicted_class}_{file.filename}"
#     output_path = os.path.join(UPLOAD_DIR, output_filename)
#     painting.save(output_path)
    
#     s3_url = upload_to_s3(userid, output_path, output_filename)
#     if s3_url:
#         print(s3_url)
#         return {
#             "status": 200,
#             "data": {
#                 "predicted_class": predicted_class,
#                 "image_url": s3_url
#             },
#             "message": "Success"
#         }
#     return {"error": "Failed to upload to S3"}