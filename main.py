import uvicorn
from server.server import app
from server.minio_client.client import MinioClient

# http://localhost:9000

if __name__ == "__main__":
    # URL = "localhost:9000"
    # USER = "guestuser"
    # PASS= "supersecret123"
    # minio_client = MinioClient(URL, USER, PASS)
    # minio_client.create_bucket("my-bucket")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
