from typing import Optional
from pydantic import BaseModel

class ASRResponse(BaseModel):
    text: str
    status: str = "success"
    processing_time_ms: Optional[float] = None

class ErrorResponse(BaseModel):
    detail: str
    status: str = "error" 