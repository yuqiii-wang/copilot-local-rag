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

class FeedbackRequest(BaseModel):
    query: str
    ai_thinking: str = ""
    ai_answer: str = ""
    user_comments: str = ""

@router.post("/feedback/accept")
async def feedback_accept(request: FeedbackRequest):
    return rag_service.process_feedback("accept", request.query, request.ai_thinking, request.ai_answer, request.user_comments)

@router.post("/feedback/reject")
async def feedback_reject(request: FeedbackRequest):
    return rag_service.process_feedback("reject", request.query, request.ai_thinking, request.ai_answer, request.user_comments)


