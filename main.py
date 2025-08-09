import uvicorn
from server.server import app
from server.minio_client.client import MinioClient

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
