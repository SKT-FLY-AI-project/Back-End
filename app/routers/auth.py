from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timezone, timedelta

from app.models import User
from app.schemas import SignupRequest, LoginRequest
from app.database import get_db
from app.services.auth_services import hash_password, verify_password, create_access_token
from app.config import settings

from fastapi import Request, Header
from fastapi.responses import RedirectResponse
import httpx
import uuid  # 추가


# config.py에서 KAKAO KEY 설정 가져오기
from app.config import KAKAO_API_KEY, KAKAO_REDIRECT_URI #, KAKAO_LOGOUT_REDIRECT_URI

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

# 카카오 로그인 URL 반환
@router.get("/kakao/loginpage", summary="카카오 로그인 페이지로 이동")
async def kakao_connect():
    url = (
        "https://kauth.kakao.com/oauth/authorize?"
        f"client_id={KAKAO_API_KEY}&redirect_uri={KAKAO_REDIRECT_URI}"
        "&response_type=code&scope=account_email"
    )
    return RedirectResponse(url=url)


# 카카오 로그인 처리 및 User 정보 반환
@router.get("/kakao/login", summary="카카오 로그인 완료 후 User 정보 및 Access Token 반환")
async def kakao_login(request: Request, code: str = None, db: AsyncSession = Depends(get_db)):
    if not code:
        raise HTTPException(status_code=400, detail="카카오 로그인 코드가 제공되지 않았습니다.")

    # 1. Access Token 발급
    token_url = "https://kauth.kakao.com/oauth/token"
    payload = {
        "grant_type": "authorization_code",
        "client_id": KAKAO_API_KEY,
        "redirect_uri": KAKAO_REDIRECT_URI,
        "code": code
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="카카오 액세스 토큰 발급 실패")

        token_data = response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="카카오 액세스 토큰이 없습니다.")

    # 2. User 정보 조회 (이메일 가져오기)
    user_info_url = "https://kapi.kakao.com/v2/user/me"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(user_info_url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="카카오 사용자 정보 조회 실패")

        user_data = response.json()

        # ✅ email 가져오기 (kakao_account.email)
        kakao_id = str(user_data["id"])
        user_email = user_data["kakao_account"].get("email")
        if not user_email:
            raise HTTPException(status_code=400, detail="사용자 이메일 정보가 제공되지 않았습니다.")

    # 3. DB에 사용자 정보 저장 (없으면 새로 추가)
    stmt = select(User).where(User.name == user_email)  # 이메일로 조회
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()  # 첫 번째 결과 가져오기

    if not user:
        user = User(
            id=str(uuid.uuid4()),
            name=user_email,   # 이메일을 name으로 저장
            password=None,  # 카카오 로그인 사용자는 비밀번호 없음
            difficulty_lv=None,
            description_st=None,
            clr_knowledge=None
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    # 4. 반환: Access Token + User 정보
    return {
        "access_token": access_token,
        "user_id": user.id,
        "user_email": user_email,   # 이메일 반환
        "message": "카카오 로그인 Access Token이 발급되었습니다."
    }




# 카카오 로그아웃
@router.get("/kakao/logout", summary="카카오 로그아웃")
async def kakao_logout(Authorization: str = Header(...)):
    logout_url = "https://kapi.kakao.com/v1/user/logout"
    headers = {
        "Authorization": f"Bearer {Authorization}"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(logout_url, headers=headers)
        if response.status_code == 200:
            return {"message": "카카오 로그아웃 성공하였습니다."}
        else:
            raise HTTPException(status_code=400, detail="카카오 로그아웃 실패하였습니다.")