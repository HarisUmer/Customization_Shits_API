import os
import io
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn
import shutil
from typing import Optional

# Import the SD generation function
from advance_wf import generate_sd_image

app = FastAPI(title="Stable Diffusion Image Generation API")

# Add CORS middleware to allow web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create required directories
os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    # Serve the index.html file from static folder
    return FileResponse("static/index.html")

@app.post("/generate/")
async def generate_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("ugly, deformed, disfigured, poor quality, low quality"),
    checkpoint_name: str = Form(r"\\SDXL-TURBO\\animexlXuebimix_v60LCM.safetensors"),
    lora_name: str = Form(r"\\SDXL\\Abstract\\logomkrdsxl.safetensors"),
    width: int = Form(1080),
    height: int = Form(1080),
    steps: int = Form(30),
    cfg: float = Form(9.0),
    sampler_name: str = Form("euler_ancestral"),
    scheduler: str = Form("normal"),
    seed: Optional[int] = Form(None)
):
    """
    Generate an image using Stable Diffusion.
    
    - Upload an image and mask
    - Provide prompt and other parameters
    - Get back the generated image
    """
    try:
        # Generate unique filenames
        image_id = str(uuid.uuid4())
        image_path = f"temp/{image_id}_image.png"
        mask_path = f"temp/{image_id}_mask.png"
        output_path = f"output/{image_id}_output.png"
        
        # Save uploaded files
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        with open(mask_path, "wb") as buffer:
            shutil.copyfileobj(mask.file, buffer)
            
        # Open images with PIL to ensure they're in the right format
        pil_image = Image.open(image_path).convert("RGB")
        pil_mask = Image.open(mask_path).convert("L")
        
        # Save the processed images
        pil_image.save(image_path)
        pil_mask.save(mask_path)
        
        # Run the generation function
        result_image = generate_sd_image(
            checkpoint_name=checkpoint_name,
            lora_name=lora_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            seed=seed if seed else int(uuid.uuid4().int % (2**32)),
            mask_dir=pil_mask,
            image_dir=pil_image,
            output_dir=output_path
        )
        
        # Save result and return it
        result_image.save(output_path)
        
        # Return the image as a streaming response
        # Save the result to a BytesIO buffer
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

# Return the image as a streaming response
        return StreamingResponse(img_byte_arr, media_type="image/png")

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image generation: {str(e)}")
    finally:
        # Clean up temporary files
        for path in [image_path, mask_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    uvicorn.run("sd_api:app", port=8000, reload=True) 