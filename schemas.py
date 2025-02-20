from pydantic import BaseModel

class DescriptionSchema(BaseModel):
    photo_id: int
    generated_text: str | None

    class Config:
        from_attributes = True