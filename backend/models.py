from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime

class OCRResult(BaseModel):
    id: Optional[int] = None
    filename: str
    ocr_text: Optional[str] = None
    image: Optional[bytes] = None
    created_at: Optional[datetime] = None
    error: Optional[str] = None

    class Config:
        from_attributes = True
