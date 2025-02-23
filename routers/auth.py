from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from models import User
from sqlalchemy import text
from database import get_db
from datetime import datetime, timezone, timedelta
from utils import hash_password, verify_password, create_access_token
from config import settings
from pydantic import BaseModel, validator
import jwt
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials


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

@router.post("/signup", summary="회원가입")
def signup(data: SignupRequest, db: Session = Depends(get_db)):
    # 같은 이름의 유저가 있는지 확인
    existing_user = db.query(User).filter(User.name == data.name).first()
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
    db.commit()
    db.refresh(new_user)
    return {"message": "User signup successfully", "user_id": new_user.id}

# @router.post("/signup", summary = "회원가입")
# def signup(name: str, password: str, db: Session = Depends(get_db)):
#     # 같은 이름의 유저가 있는지 확인
#     existing_user = db.query(User).filter(User.name == name).first()
#     if existing_user:
#         raise HTTPException(status_code=400, detail="Username already exists")

#     # 비밀번호 해싱 후 저장
#     hashed_password = hash_password(password)

#     new_user = User(
#         name=name,
#         password=hashed_password,  # 해싱된 비밀번호 저장
#         created_at=datetime.now(timezone.utc),
#         updated_at=datetime.now(timezone.utc),
#     )

#     db.add(new_user)
#     db.commit()
#     db.refresh(new_user)  # 새로 추가된 사용자 정보 반환
#     return {"message": "User signup successfully", "user_id": new_user.id}

class LoginRequest(BaseModel):
    name: str
    password: str

# @router.post("/login", summary="로그인")
# def login(data: LoginRequest, db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.name == data.name).first()
#     if not user:
#         raise HTTPException(status_code=401, detail="Invalid credentials")
    
#     # 예를 들어, 비밀번호 검증은 다음과 같이 할 수 있습니다.
#     if not verify_password(data.password, user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")

#     return {"message": "Login successful", "user_id": user.id}

@router.post("/login", summary="로그인")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.name == data.name).first()
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

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/kakao", summary = "카카오 로그인") # 아직 구현 X
def kakao_login():
    return {"message": "Kakao login successful"}


#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
# security = HTTPBearer()

# # 현재 로그인된 사용자 가져오기 아직 미완
# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     """
#     클라이언트가 전달한 JWT 토큰을 검증하고, 해당 사용자를 반환하는 함수.
#     토큰이 유효하지 않으면 401 예외를 발생시킵니다.
#     """
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         # 토큰 디코딩
#         payload = jwt.decode(
#             token,
#             settings.SECRET_KEY,           # JWT 서명 시 사용한 키
#             algorithms=[settings.ALGORITHM] # JWT 서명 시 사용한 알고리즘
#         )
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception

#     # DB에서 사용자 조회
#     user = db.query(User).filter(User.name == username).first()
#     if user is None:
#         raise credentials_exception

#     return user

# HTTPBearer 인스턴스 생성 (자동 에러 발생)
http_bearer = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
    db: Session = Depends(get_db)
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
            settings.SECRET_KEY,           # JWT 서명 시 사용한 키
            algorithms=[settings.ALGORITHM]  # JWT 서명 시 사용한 알고리즘
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # DB에서 사용자 조회
    user = db.query(User).filter(User.name == username).first()
    if user is None:
        raise credentials_exception

    return user