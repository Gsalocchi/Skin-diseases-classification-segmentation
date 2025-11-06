import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import timm

class BalancedFocalLoss(nn.Module):
    """
    Focal loss + per-class weights from counts.
    counts: list/torch.Tensor with length C (count per class in train set).
    If you don't have counts, pass None and use plain CE instead.
    """
    def __init__(self, class_counts=None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if class_counts is None:
            self.class_weights = None
        else:
            total = float(sum(class_counts))
            weights = [total / (c + 1e-6) for c in class_counts]
            w = torch.tensor(weights, dtype=torch.float32)
            w = w / w.sum() * len(weights)
            self.register_buffer("class_weights", w)

    def forward(self, logits, targets):
        if self.class_weights is None:
            ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        else:
            ce = nn.functional.cross_entropy(
                logits, targets, weight=self.class_weights, reduction="none"
            )
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma
        return (focal * ce).mean()


# -----------------------------------------------------
# MODEL
# -----------------------------------------------------
def create_model(
    num_classes: int,
    model_name: str = "vit_base_patch16_384",
    pretrained: bool = True,
) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


# -----------------------------------------------------
# TRAIN / EVAL
# -----------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, total_correct / total


# -----------------------------------------------------
# HIGH-LEVEL TRAINER
# -----------------------------------------------------
def train_model(
    train_loader,
    val_loader,
    num_classes: int = 7,
    model_name: str = "vit_base_patch16_384",
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    device: Optional[torch.device] = None,
    class_counts=None,  # pass list from your train_df if you want
    save_path: str = "best_model.pth",
):
    """
    High-level helper: trains and saves best val model.
    Returns model and a history dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes, model_name, pretrained=True).to(device)

    if class_counts is not None:
        criterion = BalancedFocalLoss(class_counts=class_counts, gamma=2.0).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # simple cosine over epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict()}, save_path)
            print(f"  ✓ New best val_acc={val_acc:.4f}, model saved to {save_path}")

    return model, history