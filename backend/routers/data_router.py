from fastapi import APIRouter, HTTPException, status, Body
from services.data_service import FrontendDataSchema, process_initial_docs, process_feedback_update, process_update_docs

router = APIRouter()

@router.post("/record_docs")
async def record_docs(data: FrontendDataSchema):
    """
    Records the initial query, referenced docs, and conversation.
    Returns the record ID.
    """
    try:
        record_id = process_initial_docs(data.model_dump())
        if not record_id:
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save data"
            )
        return {"status": "success", "id": record_id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

from fastapi import Body

@router.post("/update_docs")
async def update_docs(data: dict = Body(...)):
    """
    Updates existing record's reference docs (merge comments/keywords) using a partial payload.
    Accepts { "query": { "id": "...", "ref_docs": [...] } }
    """
    try:
        success = process_update_docs(data)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Record not found or failed to update"
            )
        return {"status": "success", "message": "Record updated"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/record_feedback")
async def record_feedback_endpoint(feedback: dict = Body(...)):
    """
    Updates the status of an existing record (accept, reject, add_confluence).
    Expects { "id": "uuid", "status": "..." }
    """
    try:
        success = process_feedback_update(feedback)
        if not success:
             raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Record not found or failed to update"
            )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
