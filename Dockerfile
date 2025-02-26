FROM nvidia/cuda:11.7-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 libgl1 git

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 후 설치
COPY requirements.txt .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade torch accelerate transformers\
RUN pip install git+https://github.com/huggingface/transformers accelerate
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 복사
COPY . .

# uvicorn으로 FastAPI 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]