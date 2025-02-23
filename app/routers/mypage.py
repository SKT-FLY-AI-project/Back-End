from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.schemas import UpdateMyPageRequest
from app.crud.user_crud import get_current_user

router = APIRouter(
    prefix="/api/user",
    tags=["User"]  
)

# 마이페이지 조회 API
@router.get("/get", summary="Get My Page")
def get_my_page(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id,
        "name": user.name,
        "difficulty_lv": user.difficulty_lv,
        "description_st": user.description_st,
        "clr_knowledge": user.clr_knowledge,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }

# 마이페이지 수정 API
@router.patch("/edit", summary="Update My Page")
def update_my_page(
    data: UpdateMyPageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if data.difficulty_lv is not None:
        user.difficulty_lv = data.difficulty_lv
    if data.description_st is not None:
        user.description_st = data.description_st
    if data.clr_knowledge is not None:
        user.clr_knowledge = data.clr_knowledge

    db.commit()
    db.refresh(user)
    return {"message": "Profile updated successfully", "user": user}