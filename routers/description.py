from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from database import get_db
from models import GeneratedDescription, UserPhoto  # UserPhoto 모델 추가
from schemas import DescriptionSchema
from services.ai_service import fetch_ai_description  # AI API 호출

router = APIRouter(prefix="/description", tags=["Description"])

@router.get("/getphoto", response_model=DescriptionSchema)
def generate_description(db: Session = Depends(get_db)):
    """최근 userphoto가 있으면 photo_id를 사용, 없으면 AI 설명을 생성 후 저장"""

    # Step 1: 가장 최근의 photo_id 가져오기 (없을 경우 None)
    result = db.execute(select(UserPhoto).order_by(UserPhoto.id.desc()).limit(1))
    latest_photo = result.scalars().first()
    photo_id = latest_photo.id if latest_photo else None  # ✅ 최근 photo_id 없으면 None

    # Step 2: AI API 호출 (설명을 AI에서 가져옴)
    description_text = fetch_ai_description(photo_id)  # ✅ AI에서 설명 생성

    if description_text:
        # Step 3: DB에 저장
        new_entry = GeneratedDescription(photo_id=photo_id, generated_text=description_text)
        db.add(new_entry)
        db.commit()
        
        return {"photo_id": photo_id, "generated_text": description_text}

    return {"message": "AI에서 설명을 가져올 수 없습니다."}