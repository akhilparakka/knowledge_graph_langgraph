from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from server.minio_client.client import MinioClient
from server.core.ingest import ingest_file
from typing import Optional
import tempfile
import time

import os
import re

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):

    try:
        url = await push_document_to_minio(file)
        print(url)
        # response = await ingest_file(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "title": title,
        "message": "Document received for processing"
    }

async def push_document_to_minio(file: UploadFile):
    URL = "localhost:9000"
    USER = "guestuser"
    PASS = "supersecret123"
    BUCKET_NAME = "documents"

    minio_client = MinioClient(URL, USER, PASS)

    if not minio_client.client.bucket_exists(BUCKET_NAME):
        minio_client.create_bucket(BUCKET_NAME)

    print(f"Existing objects in bucket '{BUCKET_NAME}':")
    for obj in minio_client.list_objects(BUCKET_NAME):
        print(f"- {obj.object_name}")

    suffix = os.path.splitext(file.filename or "")[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    object_name = _normalize_filename(file.filename or "unnamed_file")

    try:
        minio_client.upload_file(BUCKET_NAME, object_name, tmp_path)
    finally:
        os.remove(tmp_path)

    file_url = f"http://{minio_client.endpoint}/{BUCKET_NAME}/{object_name}"

    return file_url

def _normalize_filename(original_name: str, add_timestamp: bool = True, lowercase: bool = True) -> str:
    base, ext = os.path.splitext(original_name)

    if lowercase:
        base = base.lower()
    base = re.sub(r"-\d{10,}$", "", base)
    base = re.sub(r"[^\w\-]+", "_", base)
    base = re.sub(r"__+", "_", base).strip("_")

    if add_timestamp:
        base += f"-{int(time.time() * 1000)}"

    return f"{base}{ext}"
