import tensorflow as tf
from app.config import MODEL_PATH

class ModelManager:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        return self.model
    
    
from app.config import MODEL_PATH

# 모델 로드
model = tf.keras.models.load_model(MODEL_PATH)