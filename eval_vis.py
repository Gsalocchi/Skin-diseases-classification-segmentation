from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

# optional deps
try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    from sklearn.metrics import recall_score, confusion_matrix
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    data_loader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Tuple["np.ndarray", "np.ndarray", Optional[float], float]:
    """
    Run model on data_loader and return:
        y_true, y_pred, mean_loss (or None), accuracy
    """
    if np is None:
        raise RuntimeError("NumPy is required for collect_predictions but is not available.")

    model.to(device)
    model.eval()

    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        if criterion is not None:
            loss = criterion(logits, labels)
            bs = images.size(0)
            total_loss += loss.item() * bs
        else:
            bs = images.size(0)

        preds = logits.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total += bs

        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    if len(all_preds) == 0:
        raise RuntimeError("No samples in data_loader.")

    # convert to numpy
    y_pred = np.concatenate([p.numpy() for p in all_preds])  # type: ignore[arg-type]
    y_true = np.concatenate([t.numpy() for t in all_targets])  # type: ignore[arg-type]

    mean_loss: Optional[float]
    if criterion is not None and total > 0:
        mean_loss = total_loss / total
    else:
        mean_loss = None

    acc = total_correct / total if total > 0 else float("nan")

    return y_true, y_pred, mean_loss, acc


def compute_mean_recall_and_cm(
    y_true,
    y_pred,
    num_classes: Optional[int] = None,
) -> Tuple[float, "np.ndarray"]:
    """
    Compute macro/mean recall and confusion matrix.
    Uses sklearn if available, otherwise falls back to manual implementation.
    """
    if np is None:
        raise RuntimeError("NumPy is required for compute_mean_recall_and_cm but is not available.")

    if not _HAS_SKLEARN:
        # fallback: manual confusion matrix + recall
        if num_classes is None:
            num_classes = int(max(y_true.max(), y_pred.max())) + 1

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1

        # recall per class: TP / (TP + FN) = cm[i,i] / sum over row i
        recalls = []
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            denom = tp + fn
            recalls.append(tp / denom if denom > 0 else 0.0)

        mean_recall = float(sum(recalls) / len(recalls)) if len(recalls) > 0 else float("nan")
        return mean_recall, cm

    # sklearn path
    mean_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return float(mean_recall), cm


def plot_confusion_matrix_plotly(
    cm,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
):
    """
    Plot confusion matrix using Plotly (annotated heatmap).
    Returns the figure (or None if Plotly is not available).
    """
    try:
        import plotly.figure_factory as ff
    except Exception:
        print("Plotly is not available. Install plotly to see the confusion matrix.")
        return None

    if np is None:
        raise RuntimeError("NumPy is required for plot_confusion_matrix_plotly but is not available.")

    cm = np.array(cm)
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # reverse y-axis so first class is at top
    z = cm[::-1]
    y_labels = class_names[::-1]

    fig = ff.create_annotated_heatmap(
        z=z,
        x=class_names,
        y=y_labels,
        colorscale="Blues",
        showscale=True,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        xaxis=dict(constrain="domain"),
        yaxis=dict(autorange="reversed"),
    )

    fig.update_traces(
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>"
    )

    fig.show()
    return fig


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    criterion: Optional[nn.Module] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, float], "np.ndarray", Tuple["np.ndarray", "np.ndarray"]]:
    """
    High-level convenience wrapper for your vision model:
      - collects predictions (and optional loss/accuracy),
      - computes macro/mean recall and confusion matrix,
      - optionally prints metrics,
      - optionally plots confusion matrix with Plotly.

    Args:
        model: classification model (e.g. your timm ViT).
        data_loader: DataLoader yielding (images, labels).
        device: torch.device.
        class_names: optional list of class names for plotting.
        num_classes: optional number of classes for confusion matrix.
        criterion: optional loss function; if None, uses CrossEntropyLoss.
        verbose: whether to print metrics.

    Returns:
        metrics: dict with keys {"loss", "accuracy", "mean_recall"}.
        cm: confusion matrix (numpy array).
        (y_true, y_pred): numpy arrays with labels and predictions.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    y_true, y_pred, mean_loss, acc = collect_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
        criterion=criterion,
    )

    mean_recall, cm = compute_mean_recall_and_cm(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    metrics: Dict[str, float] = {
        "loss": float(mean_loss) if mean_loss is not None else float("nan"),
        "accuracy": float(acc),
        "mean_recall": float(mean_recall),
    }

    if verbose:
        print(
            f"Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Mean (macro) recall: {metrics['mean_recall']:.4f}"
        )

    # plot confusion matrix if possible
    plot_confusion_matrix_plotly(cm, class_names=class_names, title="Confusion Matrix")

    return metrics, cm, (y_true, y_pred)