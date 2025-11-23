import io
from typing import Tuple

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from .config import settings

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache: nn.Module | None = None


class MelanomaResNet(nn.Module):
    def __init__(self):
        super().__init__()
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# image transforms (keep in sync with training)
_image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225],
    ),
])


def prepare_image(file_bytes: bytes) -> Tuple[torch.Tensor, Image.Image]:
    """
    Converts raw image bytes -> (tensor [1,3,H,W], PIL image for CAM overlay)
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = _image_transform(pil_img).unsqueeze(0)  # [1,3,H,W]
    tensor = tensor.to(_device)
    return tensor, pil_img


def _load_model_from_checkpoint(path: str) -> nn.Module:
    model = MelanomaResNet()  # ← no pretrained=True here
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded model weights from {path}")
    except FileNotFoundError:
        print(f"WARNING: Model file not found at {path}, using randomly initialised ResNet18 weights.")
    except Exception as e:
        print(f"WARNING: Failed to load model from {path} ({e}); using randomly initialised ResNet18 weights.")

    model.to(_device)
    model.eval()
    return model


def get_model() -> nn.Module:
    """
    Returns a cached model instance so we don't re-load on every request.
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = _load_model_from_checkpoint(settings.MODEL_PATH)
    return _model_cache


@torch.no_grad()
def predict_with_probs(input_tensor: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """
    Runs the model and returns (logits, benign_prob, malignant_prob).
    """
    model = get_model()
    logits = model(input_tensor)  # [1, 2]

    probs = torch.softmax(logits, dim=1)
    benign = float(probs[0, 0].item())
    malignant = float(probs[0, 1].item())

    return logits, benign, malignant
