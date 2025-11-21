from fastapi import APIRouter, File, UploadFile

from .model import load_image, infer_stub
from .cam import generate_cam_stub
from .schemas import InferResponse, Probs

router = APIRouter()


@router.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)) -> InferResponse:
    raw = await file.read()
    image = load_image(raw)

    benign, malignant = infer_stub(image)
    cam_url = generate_cam_stub(image)

    return InferResponse(
        probs=Probs(benign=benign, malignant=malignant),
        camPngUrl=cam_url,
    )