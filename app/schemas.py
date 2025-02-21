from pydantic import BaseModel
from typing import List, Optional

class DescriptionSchema(BaseModel):
    photo_id: int
    generated_text: str | None

    class Config:
        from_attributes = True
        
class ChatMessage(BaseModel):
    request: str
    photo_url: str
    image_title: str
    vlm_description: Optional[str] = None
    dominant_colors: Optional[List[List[int]]] = None
    conversation_id: Optional[str] = None  # 기존 대화를 이어가기 위한 ID