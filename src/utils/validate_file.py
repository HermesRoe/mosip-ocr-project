from pathlib import Path
from fastapi import HTTPException, UploadFile
import magic  # type: ignore


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg"}
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE = 1024 * 1024 * MAX_FILE_SIZE_MB

m = magic.Magic(mime=True)


def _get_file_extensions(filename: str) -> str | None:
    if not filename:
        return None

    return Path(filename).suffix.lower().replace(".", "")


async def validate_file(file: UploadFile) -> bool:
    filename = file.filename or ""
    file_extension = _get_file_extensions(filename)

    if (not file_extension or file_extension not in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400, detail=f"Invalid file extension, Allowed: {ALLOWED_EXTENSIONS}")

    content = await file.read()

    if (len(content) > MAX_FILE_SIZE):
        await file.seek(0)
        raise HTTPException(
            status_code=413, detail=f"File size too big. Max limit {MAX_FILE_SIZE_MB} MB limit")

    detected_mime = m.from_buffer(content)  # type: ignore

    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported file type"
        )

    # Extension ↔ MIME consistency
    if detected_mime.startswith("image/") and file_extension not in {"png", "jpg", "jpeg"}:
        raise HTTPException(
            status_code=400,
            detail="File extension does not match MIME type"
        )

    # Reset file pointer for downstream usage
    await file.seek(0)
    return True
