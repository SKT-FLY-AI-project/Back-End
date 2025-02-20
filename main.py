from fastapi import FastAPI
from routers import auth, mypage, description  # routers 폴더 안의 auth.py, mypage.py를 가져옴
from database import init_db

app = FastAPI()

init_db()  # 데이터베이스 초기화

app.include_router(auth.router)  # 회원 관련 API 추가
app.include_router(mypage.router)  # 마이페이지 관련 API 추가
app.include_router(description.router)  # 설명 관련 API 추가

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)