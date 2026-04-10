from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="GenAI Vision Inference API",
    description="Asynchronous Text-to-Image serving layer built with FastAPI and PyTorch.",
    version="1.0.0"
)

# Mount the routes
app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Vision Inference ML-API"}