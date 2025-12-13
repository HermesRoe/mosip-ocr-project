from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class QualityCheck(BaseModel):
    is_blurry: bool
    blur_score: Optional[float]
    ghost_image_detected: Optional[bool] = False
    ghost_confidence: Optional[float] = None
    face_count: Optional[int] = None


class OCRField(BaseModel):
    text: str
    field: Optional[str] = None
    confidence: float
    box: List[List[int]]


class ExtractionResponse(BaseModel):
    quality_checkk: QualityCheck
    ocr_data: List[OCRField]
    heatmap_available: bool = False
    heatmap_path: Optional[str] = None
    status: str = "success"


class VerificationRequest(BaseModel):
    ocr_data: List[Dict[str, Any]]
    user_data: Dict[str, Any]


class VerifyResponse(BaseModel):
    status: str
    overall_score: float
    field_results: List[VerificationRequest]
    verified_fiels: int
