from fastapi import APIRouter
from pydantic import BaseModel
from services.rag_service import rag_service

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
    responses={404: {"description": "Not found"}},
)

class QueryRequest(BaseModel):
    query: str
    skip: int = 0
    limit: int = 5

@router.post("/retrieve")
async def retrieve(request: QueryRequest):
    return rag_service.get_dummy_search_results(request.query, skip=request.skip, limit=request.limit)


