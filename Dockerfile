# Python 베이스 이미지를 사용 (필요한 버전 선택)
FROM python:3.9-slim

RUN apt-get update && apt-get install -y git

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 후 설치
COPY requirements.txt .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade torch accelerate transformers
RUN pip install qwen-vl-utils[decord]==0.0.8
RUN pip install git+https://github.com/huggingface/transformers accelerate
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 복사
COPY . .

# uvicorn으로 FastAPI 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]