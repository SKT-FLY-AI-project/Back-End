import bcrypt
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException
import mysql.connector
from mysql.connector import pooling
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

dbconfig = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
    # "port": int(os.getenv("DB_PORT", 3306))
}

db_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=10,  # 최대 5개의 연결
    **dbconfig
)

# MySQL 연결을 가져오는 함수
def get_db():
    db = db_pool.get_connection()  # 커넥션 풀에서 연결을 가져옴
    try:
        yield db
    finally:
        db.close()
        
# @contextmanager
# def get_db():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="1234",
#         database="mydatabase"
#     )


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/query")
def query(db=Depends(get_db)):
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
    return {"results": results}

@app.post("/login")
def login(username: str, password: str, db=Depends(get_db)):
    with db.cursor(dictionary=True) as cursor:
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "user": user}

@app.post("/signup")
def signup(username: str, password: str, db=Depends(get_db)):
    with db.cursor(dictionary=True) as cursor:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        db.commit()
        cursor.close()
    
    return {"message": "Signup successful"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Flutter에서 접근 가능하도록 설정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)