from pathlib import Path
import shutil
import time
import os
import sys
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# --- PATH FIX START ---
# Force Python to look 3 levels up to find 'Opencv'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from Opencv.read import process_image 
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import Opencv. Checked path: {project_root}")
    # Temporary fallback to allow server to start even if import fails (for debugging)
    process_image = None
# --- PATH FIX END ---

from utils.validate_file import validate_file

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ExtractRouter = APIRouter(tags=["extract"])

def __save_uploaded_file(file: UploadFile) -> str:
    # Save with absolute path so CV module can find it
    file_name = f"{int(time.time() * 1000)}_{file.filename}"
    file_path = UPLOAD_DIR / file_name
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    return str(file_path.absolute())

@ExtractRouter.post("/extract")
async def extract_id(file: UploadFile = File(...)):
    if process_image is None:
        raise HTTPException(status_code=500, detail="Computer Vision module failed to load.")

    try:
        # 1. Validate File
        await validate_file(file)
        
        # 2. Save File
        file_path = __save_uploaded_file(file)
        
        # 3. CALL MEMBER A (Computer Vision)
        print(f"Sending image to CV module: {file_path}")
        cv_result = process_image(file_path)
        
        if cv_result.get("status") == "error":
            raise HTTPException(status_code=500, detail=cv_result.get("message"))

        # 4. FORMAT DATA FOR FRONTEND
        formatted_ocr_data = []
        raw_fields = cv_result.get("fields", {})
        field_boxes = cv_result.get("field_boxes", {}) 
        
        for key, value in raw_fields.items():
            # Get the raw polygon box: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            raw_box = field_boxes.get(key, None)
            
            # --- THE FIX: Convert Polygon to [x, y, w, h] ---
            final_box = [0, 0, 0, 0]
            if raw_box and len(raw_box) == 4:
                try:
                    # Flatten if it's a numpy array
                    if hasattr(raw_box, "tolist"):
                        raw_box = raw_box.tolist()
                        
                    # Calculate x, y, width, height
                    # Assuming raw_box is [top_left, top_right, bottom_right, bottom_left]
                    x = int(raw_box[0][0])
                    y = int(raw_box[0][1])
                    w = int(raw_box[2][0] - raw_box[0][0])
                    h = int(raw_box[2][1] - raw_box[0][1])
                    final_box = [x, y, w, h]
                except Exception:
                    final_box = [0, 0, 0, 0]

            formatted_ocr_data.append({
                "field": key,
                "text": value,
                "confidence": 0.95, 
                "box": final_box # Now sending [x, y, w, h]
            })

        # 5. Return Response
        return JSONResponse(content={
            "status": "success",
            "quality_check": cv_result.get("quality_check", {}),
            "ocr_data": formatted_ocr_data,
            "heatmap_path": None
        })

    except Exception as e:
        print(f"Error during extraction: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)