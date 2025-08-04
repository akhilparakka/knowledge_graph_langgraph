from fastapi import FastAPI
from server.api.routes import health
from server.api.routes.documents import router as documents_router

app = FastAPI(title="My API Server")

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(documents_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Welcome to the API Server"}
