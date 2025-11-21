from io import BytesIO
from PIL import Image
from typing import Tuple

def load_image(raw_bytes: bytes) -> Image.Image:
  return Image.open(BytesIO(raw_bytes)).convert("RGB")

def infer_stub(image: Image.Image) -> Tuple[float, float]:
  # TODO: real model later
  benign = 0.7
  malignant = 0.3
  return benign, malignant