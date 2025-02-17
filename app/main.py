from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import photo, description

app = FastAPI()
app.include_router(photo.router)
app.include_router(description.router)
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