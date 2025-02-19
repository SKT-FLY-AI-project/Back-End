from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import auth, photo, description, chat

app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     """FastAPI 실행 시 LLM 모델을 미리 로드"""
#     print("🚀 FastAPI 시작 - LLM 모델 로딩 중...")
#     llm_model.load_model()
#     print("✅ FastAPI 시작 완료!")

app.include_router(photo.router)
app.include_router(description.router)
app.include_router(chat.router)
app.mount("/static", StaticFiles(directory='./app/static'), name="static")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
