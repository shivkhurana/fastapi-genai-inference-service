from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.core.model_manager import VisionModelManager
import base64
from io import BytesIO
import asyncio

router = APIRouter()

# Initialize and load the heavy ML model once at startup
model_manager = VisionModelManager()
model_manager.load_model()

# Pydantic schema for strict data validation
class GenerationRequest(BaseModel):
    prompt: str
    steps: int = 20  # Configurable hyperparameters

@router.post("/v1/generate")
async def generate_vision_inference(request: GenerationRequest):
    try:
        # Offload the heavy PyTorch inference to a separate thread 
        # so we don't block the asynchronous event loop
        image = await asyncio.to_thread(model_manager.generate_image, request.prompt)
        
        # Convert the generated PIL image to a base64 string for the API response
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success", 
            "hyperparameters_used": {"steps": request.steps},
            "image_base64": img_str
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")