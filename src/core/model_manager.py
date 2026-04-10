import torch
from diffusers import StableDiffusionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionModelManager:
    def __init__(self):
        # Automatically detect if a GPU is available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    def load_model(self):
        logger.info(f"Loading Generative Neural Network onto {self.device}...")
        
        # Using a standard Hugging Face diffusion model
        model_id = "runwayml/stable-diffusion-v1-5"
        
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipeline = self.pipeline.to(self.device)
            
            # RESUME PROOF: Advanced memory management technique
            # This prevents Out-Of-Memory (OOM) errors during heavy inference
            self.pipeline.enable_attention_slicing()
            logger.info("Model loaded successfully with attention slicing enabled.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError("Model initialization failed.")

    def generate_image(self, prompt: str):
        if not self.pipeline:
            raise RuntimeError("Model is not loaded into memory.")
        
        logger.info(f"Generating image for prompt: '{prompt}'")
        # Run inference
        image = self.pipeline(prompt).images[0]
        return image