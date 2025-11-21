from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import CAM_DIR
from .api import router as api_router


app = FastAPI(
    title="Pigmemento ML",
    version="0.1.0",
    description="Educational melanoma recognition helper – inference + Grad-CAM stub.",
)

# Ensure CAM directory exists and serve it as static files
CAM_DIR.mkdir(exist_ok=True)
app.mount("/cams", StaticFiles(directory=str(CAM_DIR)), name="cams")

# Include the API routes (e.g. /infer)
app.include_router(api_router)


@app.get("/health")
async def health():
  return {"ok": True}