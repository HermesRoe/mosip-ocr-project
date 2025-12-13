from pathlib import Path
import shutil
import time
import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# IMPORT MEMBER A'S CODE
try:
    from Opencv.read import process_image 
except ImportError:
    # Fallback if running from different folder depth
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from Opencv.read import process_image

from utils.validate_file import validate_file

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ExtractRouter = APIRouter(tags=["extract"])

def __save_uploaded_file(file: UploadFile) -> str:
    # Save with absolute path to avoid CV errors
    file_name = f"{int(time.time() * 1000)}_{file.filename}"
    file_path = UPLOAD_DIR / file_name
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    return str(file_path.absolute())

@ExtractRouter.post("/extract")
async def extract_id(file: UploadFile = File(...)):
    try:
        # 1. Validate File
        await validate_file(file)
        
        # 2. Save File
        file_path = __save_uploaded_file(file)
        
        # 3. CALL MEMBER A (COMPUTER VISION)
        # We pass the file path to the vision module
        print(f"Processing image: {file_path}")
        cv_result = process_image(file_path)
        
        if cv_result.get("status") == "error":
            raise HTTPException(status_code=500, detail=cv_result.get("message"))

        # 4. FORMAT DATA FOR FRONTEND (Member C)
        # Member A returns "fields": {"Name": "John"}
        # Frontend needs list: [{"field": "Name", "text": "John", ...}]
        
        formatted_ocr_data = []
        raw_fields = cv_result.get("fields", {})
        
        # We try to find the matching box/confidence for each field if possible
        # (For now, we map the text directly since Member A's 'ocr_data' is raw tokens)
        for key, value in raw_fields.items():
            formatted_ocr_data.append({
                "field": key,
                "text": value,
                "confidence": 0.95, # Placeholder, as field-level confidence isn't in 'fields' dict
                "box": [0, 0, 0, 0] # Placeholder if box not explicitly linked in 'fields'
            })

        # 5. Return Response
        return JSONResponse(content={
            "status": "success",
            "quality_check": cv_result.get("quality_check", {}),
            "ocr_data": formatted_ocr_data,
            "heatmap_path": None
        })

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
