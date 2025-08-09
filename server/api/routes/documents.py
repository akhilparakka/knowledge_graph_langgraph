from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from server.minio_client.client import MinioClient
from server.rabbitmq.client import RabbitMQ
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
        rabbitmq_client = RabbitMQ()
        queue_name = "documents_to_process"
        rabbitmq_client.publish(queue_name, url)
        rabbitmq_client.close()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "title": title,
        "description": description,
        "message": "Document uploaded and sent for processing",
        "url": url
    }

async def push_document_to_minio(file: UploadFile):
    URL = "localhost:9000"
    USER = "guestuser"
    PASS = "supersecret123"
    BUCKET_NAME = "documents"

    minio_client = MinioClient(URL, USER, PASS)

    if not minio_client.client.bucket_exists(BUCKET_NAME):
        minio_client.create_bucket(BUCKET_NAME)

    object_name = _normalize_filename(file.filename or "unnamed_file")

    existing_objects = [obj.object_name for obj in minio_client.list_objects(BUCKET_NAME)]
    if object_name in existing_objects:
        print(f"Duplicate file found: {object_name}, skipping upload")
        return f"http://{minio_client.endpoint}/{BUCKET_NAME}/{object_name}"

    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        minio_client.upload_file(BUCKET_NAME, object_name, tmp_path)
    finally:
        os.remove(tmp_path)

    return f"http://{minio_client.endpoint}/{BUCKET_NAME}/{object_name}"

def _normalize_filename(original_name: str, lowercase: bool = True) -> str:
    base, ext = os.path.splitext(original_name)

    if lowercase:
        base = base.lower()
    base = re.sub(r"-\d{10,}$", "", base)
    base = re.sub(r"[^\w\-]+", "_", base)
    base = re.sub(r"__+", "_", base).strip("_")

    return f"{base}{ext}"
