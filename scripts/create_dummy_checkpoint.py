import os
import torch
from app.model import MelanomaResNet

os.makedirs("models", exist_ok=True)

model = MelanomaResNet()  # same class you use in _load_model_from_checkpoint
torch.save(model.state_dict(), "models/melanoma_resnet.pt")
print("Saved dummy checkpoint to models/melanoma_resnet.pt")