from uuid import uuid4
from PIL import Image
from .config import CAM_DIR


def center_crop(image: Image.Image) -> Image.Image:
    """Center-crop the image to a square while preserving resolution."""
    width, height = image.size
    side = min(width, height)

    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side

    return image.crop((left, top, right, bottom))


def generate_cam_stub(image: Image.Image) -> str:
    """
    Generate a fake CAM image:
    - Center-crop the original image to a square
    - Apply a simple darkening transform as a placeholder "heatmap"
    - Save it under CAM_DIR and return its URL path
    """
    # 1) Crop to a square region
    cropped = center_crop(image)

    # 2) Apply a simple transform as a placeholder heatmap
    heatmap = cropped.copy()
    heatmap = heatmap.point(lambda p: p * 0.7)

    # 3) Save the heatmap
    CAM_DIR.mkdir(exist_ok=True)
    filename = f"{uuid4()}.png"
    path = CAM_DIR / filename
    heatmap.save(path, format="PNG")

    return f"/cams/{filename}"