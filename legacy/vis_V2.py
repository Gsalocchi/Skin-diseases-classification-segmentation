from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from tqdm import tqdm


def _labels_to_indices(labels: torch.Tensor) -> torch.Tensor:
    """
    Convert labels to 1D class indices.

    Supports:
      - [B] (already indices)
      - [B, 1] (squeeze)
      - [B, C] one-hot / probabilities (argmax)
    """
    if labels.dim() == 1:
        return labels

    if labels.dim() == 2:
        if labels.size(1) == 1:  # [B, 1]
            return labels.view(-1)
        else:  # [B, C] one-hot or probabilities
            return labels.argmax(dim=1)

    # Fallback: flatten all but batch, then argmax
    return labels.view(labels.size(0), -1).argmax(dim=1)


class BalancedFocalLoss(nn.Module):
    """
    Focal loss with optional class balancing.

    Expects:
        logits: [B, C]
        targets: [B] (class indices)
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
            # normalize to keep scale reasonable
            w = w / w.sum() * len(weights)
            self.register_buffer("class_weights", w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C]
        if logits.dim() != 2:
            raise ValueError(f"BalancedFocalLoss expects logits [B, C], got {logits.shape}")

        if targets.dim() != 1:
            raise ValueError(f"BalancedFocalLoss expects 1D targets, got {targets.shape}")

        if self.class_weights is None:
            ce = F.cross_entropy(logits, targets, reduction="none")
        else:
            ce = F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                reduction="none",
            )

        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma
        return (focal * ce).mean()


class ConvViTClassifier(nn.Module):
    """
    ViT wrapper with:
      - multi-stage conv stem that gradually downsamples the image
      - final F.interpolate to match the ViT input size
      - ViT backbone that expects a fixed input size (e.g. 384x384)
      - boosted classification head

    This can accept arbitrary input spatial sizes (e.g. 600x450). The conv
    stem processes and roughly downsamples, and a final interpolation maps
    it smoothly to the ViT default input size.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "vit_base_patch16_384",
        pretrained: bool = True,
        in_channels: int = 3,
        conv_mid_channels: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Backbone (feature extractor only — no classifier)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # get features, not logits
        )

        # ViT cfg tells us the input resolution (e.g. C, 384, 384)
        input_size = self.backbone.default_cfg.get(
            "input_size", (in_channels, 224, 224)
        )
        _, h, w = input_size
        self.target_size = (h, w)

        # ---- PRE-CONV STEM ----
        # Multi-stage processing:
        #   - Stage 1: learned downsampling with stride 2
        #   - Stage 2: more convs at lower resolution
        #   - Stage 3: project back to in_channels
        # Final interpolation to self.target_size happens in forward().
        self.pre_conv = nn.Sequential(
            # Stage 1: downsample (e.g. 600x450 -> ~300x225)
            nn.Conv2d(in_channels, conv_mid_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_mid_channels),
            nn.ReLU(inplace=True),

            # Stage 2: processing at lower resolution
            nn.Conv2d(conv_mid_channels, conv_mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_mid_channels),
            nn.ReLU(inplace=True),

            # Stage 3: back to in_channels (still at downsampled size)
            nn.Conv2d(conv_mid_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # ---- EXTENDED CLASSIFICATION HEAD ----
        feat_dim = self.backbone.num_features  # dim of ViT features

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def _pool_backbone_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Ensure we end up with [B, feat_dim] from whatever forward_features returns.

        For timm ViT:
          - forward_features(x) often returns [B, tokens, dim]
        We take the mean over tokens (or you can switch to CLS token if you prefer).
        """
        if isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                # [B, tokens, dim] -> mean pool over tokens
                return feats.mean(dim=1)
            elif feats.dim() == 2:
                # already [B, dim]
                return feats
            else:
                # flatten all but batch, treat as [B, dim]
                return feats.view(feats.size(0), -1)

        # handle dict / list / tuple as in earlier versions
        if isinstance(feats, dict):
            if "cls_token" in feats:
                t = feats["cls_token"]
            elif "x" in feats:
                t = feats["x"]
            elif "pooled" in feats:
                t = feats["pooled"]
            else:
                t = next(iter(feats.values()))
            if t.dim() == 3:
                return t.mean(dim=1)
            elif t.dim() == 2:
                return t
            return t.view(t.size(0), -1)

        if isinstance(feats, (list, tuple)):
            t = feats[0]
            if t.dim() == 3:
                return t.mean(dim=1)
            elif t.dim() == 2:
                return t
            return t.view(t.size(0), -1)

        # fallback
        return feats.view(feats.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H_in, W_in] — can be e.g. 600x450
        x = self.pre_conv(x)  # [B, C, H', W'] (downsampled and processed)

        # Smooth final resize to ViT's expected resolution
        x = F.interpolate(
            x,
            size=self.target_size,   # (H_vit, W_vit) from default_cfg
            mode="bilinear",
            align_corners=False,
        )

        # ViT forward
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)

        feats = self._pool_backbone_features(feats)  # [B, feat_dim]

        logits = self.head(feats)  # [B, num_classes]
        return logits


def create_model(
    num_classes: int,
    model_name: str = "vit_base_patch16_384",
    pretrained: bool = True,
) -> nn.Module:
    """
    Baseline timm model (plain ViT with its standard classifier).
    Kept for compatibility / baseline comparisons.
    """
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def create_vis_model(
    num_classes: int,
    model_name: str = "vit_base_patch16_384",
    pretrained: bool = True,
    in_channels: int = 3,
) -> nn.Module:
    """
    New model creator that uses ViT + conv stem + boosted head.
    """
    return ConvViTClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        in_channels=in_channels,
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    total_epochs: int = 0,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # convert labels to class indices for both loss & metrics
        labels_idx = _labels_to_indices(labels)

        optimizer.zero_grad()
        logits = model(images)              # [B, num_classes]
        loss = criterion(logits, labels_idx)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs

        preds = logits.argmax(1)           # [B]
        total_correct += (preds == labels_idx).sum().item()
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
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        labels_idx = _labels_to_indices(labels)

        logits = model(images)             # [B, num_classes]
        loss = criterion(logits, labels_idx)

        bs = images.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)
        total_correct += (preds == labels_idx).sum().item()
        total += bs

    return total_loss / total, total_correct / total


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
    model: nn.Module = None,
):
    """
    High-level helper: trains and saves best val model.
    Returns model and a history dict.

    If `model` is None, a baseline timm model is created via `create_model`.
    If you want the VIS model, construct it externally with `create_vis_model`
    and pass it via the `model` argument.
    """
    # pick best device for your Mac/NVIDIA/CPU
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if model is None:
        # default: baseline ViT from timm
        model = create_model(num_classes, model_name, pretrained=True).to(device)
    else:
        model = model.to(device)

    if class_counts is not None:
        criterion = BalancedFocalLoss(class_counts=class_counts, gamma=2.0).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
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
