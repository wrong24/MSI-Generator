import base64
import io
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

# Assuming models.py with UNetGenerator is in the same directory
from models import UNetGenerator

# --- Configuration ---
# These should match the parameters used during training
MODEL_INPUT_SIZE = 224
MSI_CHANNELS = 6
SWIN_MODEL_NAME = 'swin_tiny_patch4_window7_224'
MODEL_WEIGHTS_PATH = "msi_generator_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- FastAPI App Initialization ---
app = FastAPI(title="MSI Generation API")

# --- Model Loading ---
# Load the model once when the application starts
generator = UNetGenerator(msi_channels=MSI_CHANNELS, swin_model_name=SWIN_MODEL_NAME)
generator.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device(DEVICE)))
generator.to(DEVICE)
generator.eval()

print(f"Model loaded successfully on device: {DEVICE}")

# --- Data Models for API Request/Response ---
class ImageRequest(BaseModel):
    base64_str: str

class MsiResponse(BaseModel):
    channels_base64: List[str] # List of 6 base64 encoded channel images

# --- Pre- and Post-processing Functions ---
def preprocess_image(image_pil: Image):
    """Prepares a PIL image for the model."""
    transform = transforms.Compose([
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        transforms.ToTensor()
        # Normalization is not strictly needed here if the model expects [0,1]
    ])
    return transform(image_pil).unsqueeze(0).to(DEVICE)

def postprocess_output(tensor: torch.Tensor):
    """Converts model output tensor to a list of PIL Images."""
    # Denormalize if your model outputs in [-1, 1] range (like with Tanh)
    # tensor = (tensor + 1) / 2 
    tensor = tensor.detach().squeeze(0).cpu()
    
    output_images = []
    for i in range(tensor.shape[0]):
        channel_tensor = tensor[i]
        # Convert to numpy array and scale to 0-255 for image encoding
        channel_np = (channel_tensor.numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(channel_np, mode='L') # 'L' for grayscale
        output_images.append(pil_image)
    return output_images

# --- API Endpoint ---
@app.post("/predict", response_model=MsiResponse)
async def predict(request: ImageRequest):
    """
    Receives a base64 encoded image, runs inference, and returns 6 base64 encoded MSI channels.
    """
    # 1. Decode the image
    img_bytes = base64.b64decode(request.base64_str)
    image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 2. Preprocess for the model
    input_tensor = preprocess_image(image_pil)

    # 3. Run inference
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # 4. Post-process the output
    output_pil_images = postprocess_output(output_tensor) # This will be a list of 6 PIL images

    # 5. Encode output channels to base64
    encoded_channels = []
    for img in output_pil_images:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        encoded_channels.append(encoded_img)
        
    return MsiResponse(channels_base64=encoded_channels)

@app.get("/")
def read_root():
    return {"status": "MSI Generation API is running."}