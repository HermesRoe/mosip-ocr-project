from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Dict, Any

VerifyRouter = APIRouter(tags=["verify"])

class VerifyRequest(BaseModel):
    # This matches what your api.js sends
    original: Dict[str, Any]  # The full OCR response you stored
    corrected: Dict[str, Any] # The userEdits object {"Name": "New Name"}

@VerifyRouter.post("/verify")
def verify_data(payload: VerifyRequest):
    original_data = payload.original.get("ocr_data", [])
    user_edits = payload.corrected
    
    total_fields = 0
    matches = 0
    
    # Calculate Score
    for item in original_data:
        field_name = item.get("field")
        original_text = item.get("text")
        
        # If the user edited this field, check their edit. If not, use original.
        user_text = user_edits.get(field_name, original_text)
        
        if field_name:
            total_fields += 1
            # Simple exact match check (case-sensitive)
            if original_text.strip() == str(user_text).strip():
                matches += 1
                
    score = int((matches / total_fields) * 100) if total_fields > 0 else 0
    
    return {
        "status": "success",
        "match_score": score,
        "message": "Data Verified Successfully" if score == 100 else "Data Modified by User"
    }
