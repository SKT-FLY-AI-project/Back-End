import tensorflow as tf
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from app.config import MODEL_PATH

class CNNModel:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = tf.keras.models.load_model(MODEL_PATH)

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model
    
    def __call__(self, x):
        model = self.get_model()
        return model.predict(x)
    
class LLMModel:
    def __init__(self):
        # self.model_name = "Qwen/Qwen2-VL-2B-Instruct"
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = None
        self.processor = None

    def load_model(self):
        """FastAPI 실행 시 한 번만 모델을 로드"""
        if self.model is None:
            print("🔹 LLM 모델 로드 중...")
            torch.cuda.empty_cache()
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,  # ✅ GPU는 FP16 사용, CPU는 FP32 사용
                device_map=self.device, # 원래는 auto. CPU 쓸거면 cpu로 바꿔야함.
                max_memory={0: "10GiB", "cpu": "30GiB"}
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            print("✅ LLM 모델 로드 완료!")

    def get_model(self):
        """로드된 모델을 반환"""
        if self.model is None or self.processor is None:
            self.load_model()
        return self.model, self.processor
    
# 모델 로드
cnn_model = CNNModel()
llm_model = LLMModel()