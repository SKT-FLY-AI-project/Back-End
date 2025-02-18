# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

# DATABASE_URL = "mysql+pymysql://username:password@localhost/db_name"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# 동기식
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings
from models import Base

# MySQL 연결 정보
DB_URL = settings.DATABASE_URL

# SQLAlchemy 엔진 생성
engine = create_engine(DB_URL, echo=True)

# 세션 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스 생성
Base = declarative_base()
# 테이블 생성 함수 추가
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all) 

# DB 세션 의존성 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
# from config import settings

# # MySQL 비동기 연결 정보
# DB_URL = settings.DATABASE_URL  # 예: "mysql+aiomysql://user:password@localhost/dbname"

# # 비동기 SQLAlchemy 엔진 생성
# engine = create_async_engine(DB_URL, echo=True)

# # 비동기 세션 팩토리 생성
# AsyncSessionLocal = sessionmaker(
#     autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
# )

# # Base 클래스 (모든 모델이 이 클래스를 상속해야 함)
# Base = declarative_base()

# # 비동기 테이블 생성 함수
# async def init_db():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)

# # FastAPI 의존성 주입 (비동기 세션 제공)
# async def get_async_session() -> AsyncSession:
#     async with AsyncSessionLocal() as session:
#         yield session

# from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
# from config import settings

# # ✅ `mysql+pymysql://` -> `mysql+aiomysql://` (또는 `mysql+asyncmy://`)
# DB_URL = settings.DATABASE_URL.replace("mysql+pymysql", "mysql+asyncmy")  # aiomysql 사용

# # 비동기 SQLAlchemy 엔진 생성
# engine = create_async_engine(DB_URL, echo=True)

# # 비동기 세션 팩토리 생성
# AsyncSessionLocal = sessionmaker(
#     autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
# )

# # Base 클래스 (모든 모델이 상속받아야 함)
# Base = declarative_base()

# # 비동기 테이블 생성 함수
# async def init_db():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)

# # FastAPI 의존성 주입 (비동기 세션 제공)
# async def get_async_session() -> AsyncSession:
#     async with AsyncSessionLocal() as session:
#         yield session