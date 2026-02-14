from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime

class VisionResult(BaseModel):
    id: Optional[int] = None
    filename: str
    vision_text: Optional[str] = None
    image: Optional[bytes] = None
    image_url: Optional[str] = None
    created_at: Optional[datetime] = None
    error: Optional[str] = None

    class Config:
        from_attributes = True
