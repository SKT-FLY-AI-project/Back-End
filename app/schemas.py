from pydantic import BaseModel, validator
from typing import List, Optional

class SignupRequest(BaseModel):
    name: str
    password: str

    @validator('name', 'password')
    def no_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("빈 칸은 허용되지 않습니다.")
        return v.strip()

class LoginRequest(BaseModel):
    name: str
    password: str
        
class UpdateMyPageRequest(BaseModel):
    difficulty_lv: Optional[str] = None
    description_st: Optional[str] = None
    clr_knowledge: Optional[str] = None
    
class ChatMessage(BaseModel):
    request: str
    image_url: str
    title: str
    artist: str
    rich_description: Optional[str] = None
    dominant_colors: Optional[List[List[int]]] = None
    conversation_id: Optional[str] = None  # 기존 대화를 이어가기 위한 ID