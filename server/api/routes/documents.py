from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "title": title,
        "message": "Document received for processing"
    }
