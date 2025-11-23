import io
import os
import uuid

import boto3
from botocore.client import Config
import numpy as np
import torch
from botocore.exceptions import ClientError
from torch import nn
from PIL import Image

from .config import settings
from .model import get_model, _device

r2_client = None

class GradCAM:
    def __init__(self, target_layer_name: str | None = None):
        """
        Grad-CAM implementation that:
        - Uses the given target_layer_name if it exists
        - Otherwise falls back to the last Conv2d in the model
        - If no Conv2d is found, returns a uniform heatmap instead of crashing
        """
        self.model = get_model()
        self.model.eval()

        modules = dict(self.model.named_modules())
        self.target_layer = None

        # Try explicit target_layer_name first
        if target_layer_name is not None and target_layer_name in modules:
            self.target_layer = modules[target_layer_name]
            print(f"GradCAM using explicit target layer: {target_layer_name}")
        else:
            # Fallback: last Conv2d in the network
            last_conv = None
            last_name = None
            for name, module in modules.items():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
                    last_name = name

            if last_conv is not None:
                self.target_layer = last_conv
                print(f"GradCAM fallback to last Conv2d layer: {last_name}")
            else:
                # No conv layer at all (e.g. DummyNet)
                print(
                    "WARNING: No Conv2d layers found in model; "
                    "Grad-CAM will return a uniform heatmap."
                )

        self.activations = None
        self.gradients = None

        if self.target_layer is not None:
            def fwd_hook(module, inp, out):
                self.activations = out.detach()

            def bwd_hook(module, grad_in, grad_out):
                self.gradients = grad_out[0].detach()

            self.target_layer.register_forward_hook(fwd_hook)
            self.target_layer.register_backward_hook(bwd_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        input_tensor: [1,3,H,W]
        returns: heatmap [H,W] in [0,1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # [1, num_classes]
        score = output[0, class_idx]

        # If we have no conv layer, just return a uniform heatmap
        if self.target_layer is None:
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            cam = np.ones((H, W), dtype=np.float32)
            return cam

        score.backward()

        # [B, C, H, W]
        gradients = self.gradients  # dY/dA
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = torch.relu(cam)
        cam = cam[0, 0].cpu().numpy()

        # normalize to 0–1
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam

def get_r2_client():
    global r2_client
    if r2_client is None:
        if not all([
            settings.R2_ACCOUNT_ID,
            settings.R2_ACCESS_KEY_ID,
            settings.R2_SECRET_ACCESS_KEY,
        ]):
            return None

        endpoint_url = f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

        r2_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )
    return r2_client


def _build_overlay_image(cam, base_image: Image.Image) -> Image.Image:
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(
        base_image.size, resample=Image.BILINEAR
    )

    cam_arr = np.array(cam_img)
    heatmap = np.zeros((*cam_arr.shape, 4), dtype=np.uint8)
    heatmap[..., 0] = cam_arr
    heatmap[..., 3] = (cam_arr * 0.7).astype(np.uint8)

    heatmap_img = Image.fromarray(heatmap, mode="RGBA")

    base_rgba = base_image.convert("RGBA")
    overlay = Image.alpha_composite(base_rgba, heatmap_img)
    return overlay


def _save_cam_overlay(cam, base_image: Image.Image) -> str:
    filename = f"{uuid.uuid4()}.png"
    overlay = _build_overlay_image(cam, base_image)

    client = get_r2_client()
    if client and settings.R2_BUCKET and settings.R2_PUBLIC_BASE_URL:
        try:
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            buf.seek(0)

            key = f"cams/{filename}"
            client.put_object(
                Bucket=settings.R2_BUCKET,
                Key=key,
                Body=buf,
                ContentType="image/png",
            )

            cam_url = f"{settings.R2_PUBLIC_BASE_URL}/{key}"
            print(f"Uploaded CAM to R2: {cam_url}")
            return cam_url
        except ClientError as e:
            print(f"R2 upload failed, falling back to local storage: {e}")

    # Fallback: local storage
    os.makedirs(settings.CAM_OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(settings.CAM_OUTPUT_DIR, filename)
    overlay.save(filepath)
    cam_url = f"{settings.BASE_URL}/cams/{filename}"
    print(f"Saved CAM locally: {cam_url}")
    return cam_url


def generate_cam(
    input_tensor,
    base_image: Image.Image,
    class_idx: int,
    target_layer_name: str | None = None,
) -> str:
    grad_cam = GradCAM(target_layer_name=target_layer_name)
    cam = grad_cam(input_tensor, class_idx=class_idx)
    return _save_cam_overlay(cam, base_image)