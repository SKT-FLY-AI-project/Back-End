from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from models import User
from sqlalchemy import text
from database import get_db
from datetime import datetime, timezone
from utils import hash_password, verify_password
import jwt
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials


router = APIRouter(
    prefix="/api/auth",
    tags=["Auth"]
)

@router.post("/signup", summary = "회원가입")
def signup(name: str, password: str, db: Session = Depends(get_db)):
    # 같은 이름의 유저가 있는지 확인
    existing_user = db.query(User).filter(User.name == name).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    # 비밀번호 해싱 후 저장
    hashed_password = hash_password(password)

    new_user = User(
        name=name,
        password=hashed_password,  # 해싱된 비밀번호 저장
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)  # 새로 추가된 사용자 정보 반환
    return {"message": "User signup successfully", "user_id": new_user.id}

@router.post("/login", summary = "로그인")
def login(name: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.name == name).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # 비밀번호 검증
    if not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"message": "Login successful", "user_id": user.id}

@router.get("/kakao", summary = "카카오 로그인") # 아직 구현 X
def kakao_login():
    return {"message": "Kakao login successful"}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# token 가져오기 아직 미완
@router.post("/token", summary="Get Access Token")
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.name == form_data.username).first()
    if not user or not form_data.password == user.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # JWT 토큰 생성
    token_data = {
        "sub": str(user.id),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

# 현재 로그인된 사용자 가져오기 아직 미완
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")

        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")