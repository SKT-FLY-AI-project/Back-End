from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timezone, timedelta

from app.models import User
from app.schemas import SignupRequest, LoginRequest
from app.database import get_db
from app.services.auth_services import hash_password, verify_password, create_access_token
from app.config import settings

router = APIRouter(
    prefix="/api/auth",
    tags=["Auth"]
)

@router.post("/signup", summary="회원가입")
async def signup(data: SignupRequest, db: AsyncSession = Depends(get_db)):
    # 같은 이름의 유저가 있는지 확인
    result = await db.execute(select(User).where(User.name == data.name))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    # 비밀번호 해싱 후 저장
    hashed_password = hash_password(data.password)

    new_user = User(
        name=data.name,
        password=hashed_password,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return {"message": "User signup successfully", "user_id": new_user.id}

@router.post("/login", summary="로그인")
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.name == data.name))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # JWT 토큰 생성
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.name},
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "user_name": user.name
    }

@router.get("/kakao", summary="카카오 로그인")  # 아직 구현 X
async def kakao_login():
    return {"message": "Kakao login successful"}