import os

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .schemas import InferResponse, Probs
from .model import prepare_image, predict_with_probs
from .grad_cam import generate_cam

app = FastAPI(title="Pigmemento ML Service")

# CORS – allow API + app + local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://localhost:19006",
        "https://pigmemento.app",
        "https://www.pigmemento.app",
        "https://api.pigmemento.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve CAM images (local fallback and/or primary storage)
os.makedirs(settings.CAM_OUTPUT_DIR, exist_ok=True)
app.mount("/cams", StaticFiles(directory=settings.CAM_OUTPUT_DIR), name="cams")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    # Basic validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read file bytes
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Load + preprocess
        tensor, pil_img = prepare_image(image_bytes)

        # Prediction
        logits, benign, malignant = predict_with_probs(tensor)

        # Choose class_idx for CAM: 0=benign, 1=malignant
        class_idx = int(torch.argmax(logits, dim=1).item())

        # Grad-CAM overlay (R2 or local handled inside generate_cam)
        cam_url = generate_cam(
            input_tensor=tensor,
            base_image=pil_img,
            class_idx=class_idx,
            # target_layer_name="backbone.layer4",  # uncomment if your GradCAM uses this
        )

        return InferResponse(
            probs=Probs(
                benign=benign,
                malignant=malignant,
            ),
            camPngUrl=cam_url,
        )

    except HTTPException:
        # Re-raise expected HTTP errors
        raise
    except Exception as e:
        # Catch-all for unexpected issues
        print(f"[ERROR] /infer failed: {e}")
        raise HTTPException(status_code=500, detail="Inference failed.")