from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Union, Dict
from services.download_service import download_service

router = APIRouter(
    prefix="/download",
    tags=["download"],
    responses={404: {"description": "Not found"}},
)

class DownloadRequest(BaseModel):
    # Accept either a list of URL strings or a list of objects with { url: str, comment: Optional[str] }
    urls: Union[List[Dict[str, Optional[str]]], List[str]]
    query: Optional[str] = None

@router.post("/fetch")
async def fetch_content(request: DownloadRequest):
    return download_service.fetch_content(request.urls, request.query)
