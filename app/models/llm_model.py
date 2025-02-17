import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

class LLMModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load_model(self):
        """FastAPI 실행 시 한 번만 모델을 로드"""
        if self.model is None:
            print("🔹 LLM 모델 로드 중...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            print("✅ LLM 모델 로드 완료!")

    def get_model(self):
        """로드된 모델을 반환"""
        if self.model is None or self.processor is None:
            self.load_model()
        return self.model, self.processor

# 모델 인스턴스 생성 및 로드
llm_model = LLMModel()
llm_model.load_model()  # FastAPI 실행 시 모델을 한 번 로드