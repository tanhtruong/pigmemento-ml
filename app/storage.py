from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional
from uuid import uuid4

from PIL import Image

from .config import settings

try:
    import boto3
except ImportError:
    boto3 = None


def _r2_configured() -> bool:
    return all(
        [
            settings.R2_ACCOUNT_ID,
            settings.R2_ACCESS_KEY_ID,
            settings.R2_SECRET_ACCESS_KEY,
            settings.R2_BUCKET,
            settings.R2_PUBLIC_BASE_URL,
        ]
    )


def save_cam_and_get_url(cam_image: Image.Image) -> str:
    """
    Saves a CAM image either:
      - to Cloudflare R2 (if configured), or
      - to local disk under CAM_OUTPUT_DIR

    Returns a public URL that the app can use.
    """
    cam_id = f"{uuid4()}.png"

    if _r2_configured():
        if boto3 is None:
            raise RuntimeError(
                "boto3 is required for R2 uploads but is not installed."
            )

        # Prepare in-memory bytes
        buf = BytesIO()
        cam_image.save(buf, format="PNG")
        buf.seek(0)

        endpoint_url = f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        )

        key = f"cams/{cam_id}"

        s3.upload_fileobj(
            buf,
            settings.R2_BUCKET,
            key,
            ExtraArgs={"ContentType": "image/png", "ACL": "public-read"},
        )

        base = settings.R2_PUBLIC_BASE_URL.rstrip("/")
        return f"{base}/cams/{cam_id}"

    # Fallback: local filesystem
    out_dir = Path(settings.CAM_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_path = out_dir / cam_id
    cam_image.save(cam_path)

    base = settings.BASE_URL.rstrip("/")
    # We’ll serve this with FastAPI static files mounted at /cams
    return f"{base}/cams/{cam_id}"