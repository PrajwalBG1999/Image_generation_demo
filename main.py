from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import torch
import random
from pydantic import BaseModel
from typing import List
import einops
import base64
import os
import uuid

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

app = FastAPI()

# Ensure an output directory exists to save images and create one if doesn't exist
OUTPUT_FOLDER = "output"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize the Canny detector and model
apply_canny = CannyDetector()
model = create_model('./models/cldm_v15.yaml').cpu()

# Select the device based upon availability
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load the model
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location=device))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# Calculate low and high threshold using adaptive thresholding (Extra added feature)
def adaptive_threshold(image: np.ndarray) -> tuple:
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray_image)
    std_dev = np.std(gray_image)
    low_threshold = max(0, mean - std_dev)
    high_threshold = min(255, mean + std_dev)
    return int(low_threshold), int(high_threshold)

# Request data model
class ProcessRequest(BaseModel):
    prompt: str
    a_prompt: str
    n_prompt: str
    num_samples: int
    image_resolution: int
    ddim_steps: int
    guess_mode: bool
    strength: float
    scale: float
    seed: int
    eta: float

# Helper function to convert an image (NumPy array) to a Base64 string.
def image_to_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise Exception("Could not encode image!")
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

# Helper function to save an image to disk.
def save_image(image: np.ndarray, prefix: str = "image") -> str:
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(filepath, image)
    return filepath
    
@app.post("/process/")
async def process_image(file: UploadFile = File(...), request: str = Form(...)):
    """
    Endpoint to process an uploaded image file.

    Parameters:
    - file: The image file to be processed, uploaded by the user.
    - request: A string containing additional information or parameters for processing the image.

    This asynchronous function handles POST requests to the "/process/" route,
    allowing users to upload an image file along with a form string for processing.
    """

    # Parse the JSON request data
    request_data = ProcessRequest.parse_raw(request)
    
    # Read and decode the uploaded image
    input_image = np.frombuffer(await file.read(), np.uint8)
    input_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)
    
    with torch.no_grad():
        # Resize the image
        img = resize_image(HWC3(input_image), request_data.image_resolution)
        H, W, C = img.shape

        low_threshold, high_threshold = adaptive_threshold(img)

        # Apply Canny edge detection
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        # Create a control tensor from the detected map
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(request_data.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # Set the seed (random if -1)
        if request_data.seed == -1:
            request_data.seed = random.randint(0, 65535)
        seed_everything(request_data.seed)

        # Prepare conditioning dictionaries for the model
        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([request_data.prompt + ', ' + request_data.a_prompt] * request_data.num_samples)]
        }
        un_cond = {
            "c_concat": None if request_data.guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([request_data.n_prompt] * request_data.num_samples)]
        }
        shape = (4, H // 8, W // 8)

        # Adjust control scales depending on guess mode
        if request_data.guess_mode:
            model.control_scales = [request_data.strength * (0.825 ** float(12 - i)) for i in range(13)]
        else:
            model.control_scales = [request_data.strength] * 13

        # Run the DDIM sampler
        samples, intermediates = ddim_sampler.sample(
            request_data.ddim_steps,
            request_data.num_samples,
            shape,
            cond,
            verbose=False,
            eta=request_data.eta,
            unconditional_guidance_scale=request_data.scale,
            unconditional_conditioning=un_cond
        )

        # Decode the generated samples
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # Convert the detected map and generated images to Base64 strings for visualization
        detected_map_b64 = image_to_base64(detected_map)
        results_b64 = [image_to_base64(x_samples[i]) for i in range(request_data.num_samples)]
        
        # Save the images to disk
        detected_map_path = save_image(detected_map, prefix="detected_map")
        results_paths = [save_image(x_samples[i], prefix="result") for i in range(request_data.num_samples)]
    
    # Return the Base64 images and saved file paths
    return JSONResponse(content={
        "detected_map": detected_map_b64,
        "results": results_b64,
        "saved_files": {
            "detected_map": detected_map_path,
            "results": results_paths
        }
    })

@app.get("/")
async def read_index():
    """
    Endpoint to serve the main index page.

    This asynchronous function handles GET requests to the root ("/") route
    and returns the 'index.html' file as a response. This is typically the
    entry point of the web application, providing the initial user interface.
    """
    return FileResponse('index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)