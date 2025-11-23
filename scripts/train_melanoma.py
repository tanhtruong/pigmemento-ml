import os
import argparse
from pathlib import Path
from typing import Tuple
import sys

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.model import MelanomaResNet  # re-use your API model class

def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Train/val transforms – keep them in sync with app.model._image_transform
    so inference matches training.
    """
    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # You can add light augmentation here:
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfm, val_tfm


def build_dataloaders(data_root: str, batch_size: int, num_workers: int = 4):
    """
    Expects:
      data_root/
        train/
          benign/
          malignant/
        val/
          benign/
          malignant/
    """
    train_tfm, val_tfm = build_transforms()

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfm)

    print("Class mapping (ImageFolder):", train_ds.class_to_idx)  # expect {'benign': 0, 'malignant': 1}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1) forward
        logits = model(images)  # [B, 2]

        # 2) compute loss
        loss = criterion(logits, targets)

        # 3) backward (compute gradients)
        optimizer.zero_grad()
        loss.backward()

        # 4) gradient descent step (update weights)
        optimizer.step()

        # bookkeeping
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == targets).sum().item()
        total += images.size(0)

        if (step + 1) % 20 == 0:
            print(
                f"Epoch {epoch} | Step {step+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="Path to data folder with train/ and val/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="models/melanoma_resnet.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data
    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = MelanomaResNet()
    model.to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"✅ New best model saved to {out_path} (val acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val acc = {best_val_acc:.4f}")
    print(f"Final weights saved (best) at: {out_path}")


if __name__ == "__main__":
    main()