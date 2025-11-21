from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from pathlib import Path
from typing import Dict

from PIL import Image

app = FastAPI()

# Directory where we store CAM images
CAM_DIR = Path("cams")
CAM_DIR.mkdir(exist_ok=True)

# Serve /cams/* as static files
app.mount("/cams", StaticFiles(directory=str(CAM_DIR)), name="cams")

# CORS is not strictly needed if only your .NET API calls this,
# but it's fine to allow localhost for testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Minimal stub:
    - Accepts an image upload
    - Returns dummy probabilities
    - Writes out a fake CAM PNG and returns its URL
    """

    # Read file into PIL image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # TODO: replace this with your real model inference:
    benign_prob = 0.7
    malignant_prob = 0.3

    # For now, create a "fake CAM" by just darkening the image or something simple
    # (so you visually know it's "the CAM", but it's not real)
    heatmap = image.copy().resize((256, 256))
    heatmap = heatmap.point(lambda p: p * 0.7)  # simple darkening as a placeholder

    cam_filename = f"{uuid4()}.png"
    cam_path = CAM_DIR / cam_filename
    heatmap.save(cam_path, format="PNG")

    cam_url = f"/cams/{cam_filename}"

    return {
        "probs": {
            "benign": benign_prob,
            "malignant": malignant_prob,
        },
        "camPngUrl": cam_url,
    }