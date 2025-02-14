from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Flutter에서 접근 가능하도록 설정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)