from typing import Optional, List, Tuple
import torch
from tqdm.auto import tqdm

# optional deps
try:
    import numpy as np
    from sklearn.metrics import recall_score, confusion_matrix
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Run model on data_loader and return (y_true, y_pred) as numpy arrays.
    """
    model.to(device)
    model.eval()

    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for xb, yb in tqdm(data_loader, desc="Collecting predictions", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)

        out = model(xb)
        preds = out.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())

    if len(all_preds) == 0:
        raise RuntimeError("No samples in data_loader.")

    # convert to numpy
    if hasattr(all_preds[0], "numpy"):
        y_pred = np.concatenate([p.numpy() for p in all_preds])
        y_true = np.concatenate([t.numpy() for t in all_targets])
    else:
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

    return y_true, y_pred


def compute_mean_recall_and_cm(
    y_true,
    y_pred,
    num_classes: Optional[int] = None,
) -> Tuple[float, "np.ndarray"]:
    """
    Compute macro/mean recall and confusion matrix.
    Uses sklearn if available, otherwise falls back to manual implementation.
    """
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
    if not _HAS_PLOTLY:
        print("Plotly is not available. Install plotly to see the confusion matrix.")
        return None

    import numpy as np  # safe here; already needed above

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
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[float, "np.ndarray", Tuple["np.ndarray", "np.ndarray"]]:
    """
    Convenience wrapper:
      - collects predictions,
      - computes macro/mean recall and confusion matrix,
      - optionally prints mean recall,
      - optionally plots confusion matrix with Plotly.

    Returns:
        mean_recall, cm, (y_true, y_pred)
    """
    y_true, y_pred = collect_predictions(model, data_loader, device)
    mean_recall, cm = compute_mean_recall_and_cm(
        y_true, y_pred, num_classes=num_classes
    )

    if verbose:
        print(f"Mean (macro) recall: {mean_recall:.4f}")

    # plot confusion matrix if possible
    plot_confusion_matrix_plotly(cm, class_names=class_names, title="Confusion Matrix")

    return mean_recall, cm, (y_true, y_pred)
