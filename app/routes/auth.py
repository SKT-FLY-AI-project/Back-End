from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models import User
from app.database import get_db
from datetime import datetime, timezone, timedelta
from app.utils.utils import hash_password, verify_password, create_access_token
from app.config import settings
from pydantic import BaseModel, validator
import jwt

router = APIRouter(
    prefix="/api/auth",
    tags=["Auth"]
)

SECRET_KEY = settings.SECRET_KEY

class SignupRequest(BaseModel):
    name: str
    password: str

    @validator('name', 'password')
    def no_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("빈 칸은 허용되지 않습니다.")
        return v.strip()

class LoginRequest(BaseModel):
    name: str
    password: str

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

http_bearer = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
    db: AsyncSession = Depends(get_db)
):
    """
    클라이언트가 전달한 JWT 토큰(Authorization 헤더)을 검증하고, 
    해당 사용자를 반환하는 함수.
    토큰이 유효하지 않으면 401 예외를 발생시킵니다.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # HTTPBearer를 통해 받은 credentials에서 실제 토큰을 추출
    token = credentials.credentials

    try:
        # 토큰 디코딩
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception

    result = await db.execute(select(User).where(User.name == username))
    user = result.scalars().first()
    if user is None:
        raise credentials_exception

    return user
