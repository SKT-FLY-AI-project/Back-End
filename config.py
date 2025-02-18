from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings()