from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
import models
from pgdb.pgdb_manager import pg_manager
from services import ocr_service

router = APIRouter(
    prefix="/ocr",
    tags=["ocr"],
    responses={404: {"description": "Not found"}},
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/", response_model=models.OCRResult)
async def create_ocr_item(file: UploadFile = File(...)):
    # 1. Save the uploaded file locally (temporarily)
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Run PaddleOCR on the image and save results via service
        extracted_text, new_id, created_at = ocr_service.extract_text_from_image(file_path)
        
        if new_id:
            return models.OCRResult(
                id=new_id, 
                filename=file.filename, 
                ocr_text=extracted_text,
                created_at=created_at
            )
        else:
            return models.OCRResult(
                id=-1,
                filename=file.filename,
                ocr_text=extracted_text,
                error="Failed to save OCR result"
            )
        
    except Exception as e:
        print(f"General processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{item_id}", response_model=models.OCRResult)
def read_ocr_item(item_id: int):
    query = "SELECT id, filename, ocr_text, created_at FROM repo_ask.ocr_results WHERE id = %s"
    result = pg_manager.execute_query(query, (item_id,), fetch=True)
    
    if not result:
        raise HTTPException(status_code=404, detail="Item not found")
        
    row = result[0]
    return models.OCRResult(
        id=row[0],
        filename=row[1],
        ocr_text=row[2],
        created_at=row[3]
    )
