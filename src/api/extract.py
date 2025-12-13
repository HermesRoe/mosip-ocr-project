from pathlib import Path
import shutil
import time
from typing import Any
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import magic  # type:ignore
from utils.validate_file import validate_file  # type: ignore

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

magic = magic.Magic(mime=True)


ExtractRouter = APIRouter(
    tags=["extract"]
)

# TODO : Call CV
def __save_uploaded_file(file: UploadFile) -> Path:
    file_name = f"{int(time.time() * 1000)}_{file.filename}"
    file_path = UPLOAD_DIR / file_name

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return file_path


@ExtractRouter.post("/extract")
async def extract_id(file: UploadFile = File(...)) -> JSONResponse:
    try:
        file_validate = await validate_file(file)
        await file.seek(0)

        file_path = __save_uploaded_file(file)

        await file.seek(0)
        content = await file.read()

        response: dict[str, Any] = {"filename": file.filename, "size": len(
            content), "detected_type": file.content_type, "status": f"{file_validate}", "uploaded_directory": str(file_path)}
        return JSONResponse(content=response)
    except Exception as e:
        print(e)
        return JSONResponse(content=f"error {e}")
