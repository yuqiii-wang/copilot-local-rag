from fastapi import APIRouter
from pydantic import BaseModel
from services.download_service import download_service

router = APIRouter(
    prefix="/download",
    tags=["download"],
    responses={404: {"description": "Not found"}},
)

class DownloadRequest(BaseModel):
    urls: list[str]

@router.post("/fetch")
async def fetch_content(request: DownloadRequest):
    return download_service.fetch_content(request.urls)
