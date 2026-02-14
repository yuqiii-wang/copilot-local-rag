from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
import models
from services import vision_service

router = APIRouter(
    prefix="/vision",
    tags=["vision"],
    responses={404: {"description": "Not found"}},
)

# Store uploaded files under uploads/images so they are served at /uploads/images/...
UPLOAD_DIR = os.path.join("uploads", "images")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=models.VisionResult)
async def create_vision_item(file: UploadFile = File(...)):
    """Save uploaded image and return minimal metadata.

    This endpoint no longer performs OCR/Vision analysis immediately. The frontend can use the
    returned metadata to send the image into the chat flow.
    """
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        meta = vision_service.register_uploaded_image(file_path, file.filename)

        return models.VisionResult(
            id=meta.get("id", -1),
            filename=meta.get("filename", file.filename),
            vision_text="",  # Formerly ocr_text
            image_url=meta.get("image_url"),
            created_at=meta.get("created_at")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/delete')
async def delete_uploaded_image(payload: dict):
    """Delete an uploaded image file. Expects JSON: {"filename": "name.jpg"}

    Returns {deleted: true/false}
    """
    try:
        filename = payload.get('filename')
        if not filename:
            raise HTTPException(status_code=400, detail='filename is required')

        # Only allow basename to avoid directory traversal
        base = os.path.basename(filename)
        file_path = os.path.join('uploads', 'images', base)

        if os.path.exists(file_path):
            os.remove(file_path)
            return {"deleted": True}
        else:
            return {"deleted": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


