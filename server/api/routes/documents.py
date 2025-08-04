from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from server.core.ingest import ingest_file

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):

    try:
        response = await ingest_file(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "title": title,
        "message": "Document received for processing"
    }
