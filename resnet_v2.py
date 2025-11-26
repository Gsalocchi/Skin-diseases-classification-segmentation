from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm



class BalancedFocalLoss(nn.Module):
    """
    Focal loss with optional class balancing based on class counts.

    Args:
        class_counts: list/sequence of counts for each class in the training set.
        gamma: focal loss focusing parameter.
    """

    def __init__(self, class_counts: Optional[List[int]] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if class_counts is None:
            self.class_weights = None
        else:
            total = float(sum(class_counts))
            weights = [total / (c + 1e-6) for c in class_counts]
            w = torch.tensor(weights, dtype=torch.float32)
            # normalize to keep scale reasonable
            w = w / w.sum() * len(weights)
            self.register_buffer("class_weights", w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights is None:
            ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        else:
            ce = nn.functional.cross_entropy(
                logits, targets, weight=self.class_weights, reduction="none"
            )
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma
        return (focal * ce).mean()


# ============================================================
# Model: Learned Resizer + ResNet50 backbone + Head
# ============================================================

class LearnedResizer(nn.Module):
    """
    A small conv stack + interpolation that maps 3xHxW images
    (e.g. 3x400x650) to 3x224x224 for the pretrained backbone.

    This is learned jointly with the rest of the network.
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        target_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.target_size = target_size

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # downsample a bit
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # project back to 3 channels so it still looks like an image
        self.proj = nn.Conv2d(mid_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] (e.g. H=400, W=650)
        x = self.conv_block(x)   # [B, mid, H', W']
        x = self.proj(x)         # [B, 3, H', W']

        # fixed resize to 224x224 for the pretrained backbone
        x = F.interpolate(
            x,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        return x


class ResNet50WithResizerAndHead(nn.Module):
    """
    Full model:
        3x400x650 image
            -> LearnedResizer -> 3x224x224
            -> ResNet-50 backbone (pretrained)
            -> MLP head -> num_classes logits
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        backbone_name: str = "resnet50",
        head_hidden_dim: int = 512,
        head_dropout: float = 0.3,
    ):
        super().__init__()

        # 1) learned resizer: arbitrary HxW -> 224x224
        self.resizer = LearnedResizer(
            in_channels=3,
            mid_channels=32,
            target_size=(224, 224),
        )

        # 2) pretrained backbone at 224x224
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,       # no classifier
            global_pool="avg",   # returns [B, C]
        )
        in_features = self.backbone.num_features

        # 3) extra head + classification
        self.head = nn.Sequential(
            nn.Linear(in_features, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] (e.g. 3x400x650)
        x = self.resizer(x)          # -> [B, 3, 224, 224]
        feats = self.backbone(x)     # -> [B, C]
        logits = self.head(feats)    # -> [B, num_classes]
        return logits


def create_resnet_model(
    num_classes: int = 7,
    pretrained: bool = True,
    backbone_name: str = "resnet50",
    head_hidden_dim: int = 512,
    head_dropout: float = 0.3,
) -> nn.Module:
    """
    Convenience factory for the ResNet50WithResizerAndHead model.
    """
    return ResNet50WithResizerAndHead(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone_name=backbone_name,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
    )


# ============================================================
# Device helper (MPS-aware)
# ============================================================

def get_device() -> torch.device:
    """
    Pick best available device (MPS, CUDA, then CPU).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Training / evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    total_epochs: int = 0,
) -> Tuple[float, float]:
    """
    Train for one epoch. Returns (avg_loss, avg_accuracy).
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total += bs

        avg_loss = total_loss / total
        avg_acc = total_correct / total
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate on a validation/test loader.
    Returns (avg_loss, avg_accuracy).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        bs = images.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total += bs

    return total_loss / total, total_correct / total


def train_model(
    train_loader,
    val_loader,
    num_classes: int = 7,
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    device: Optional[torch.device] = None,
    class_counts: Optional[List[int]] = None,
    save_path: str = "best_resnet_model.pth",
    model: Optional[nn.Module] = None,
    backbone_name: str = "resnet50",
    head_hidden_dim: int = 512,
    head_dropout: float = 0.3,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    High-level training helper. Trains and saves the best validation model.

    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        num_classes: number of output classes (e.g. 7).
        epochs: number of epochs.
        lr: learning rate.
        weight_decay: weight decay for AdamW.
        device: torch.device or None (auto-pick).
        class_counts: optional list of class counts for BalancedFocalLoss.
        save_path: where to save best model.
        model: optional pre-constructed model. If None, creates ResNet50WithResizerAndHead.
        backbone_name: timm backbone name (default: "resnet50").
        head_hidden_dim: hidden units in the head MLP.
        head_dropout: dropout probability in the head.

    Returns:
        model: trained model (with weights from the last epoch, not re-loaded best).
        history: dict with keys "train_loss", "train_acc", "val_loss", "val_acc".
    """
    if device is None:
        device = get_device()

    if model is None:
        model = create_resnet_model(
            num_classes=num_classes,
            pretrained=True,
            backbone_name=backbone_name,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
        )

    model = model.to(device)

    if class_counts is not None:
        criterion = BalancedFocalLoss(class_counts=class_counts, gamma=2.0).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict()}, save_path)
            print(f"  ✓ New best val_acc={val_acc:.4f}, model saved to {save_path}")

    return model, history

