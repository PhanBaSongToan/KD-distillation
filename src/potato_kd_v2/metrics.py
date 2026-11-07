"""Metrics, test-time augmentation and post-hoc logit adjustment."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def confusion_to_metrics(confusion: np.ndarray, classes: List[str]) -> Dict:
    n = confusion.shape[0]
    f1s, recalls, precisions = [], [], []
    for c in range(n):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / max(p + r, 1e-12))
    total = confusion.sum()
    return {
        "accuracy": float(np.trace(confusion) / max(total, 1)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.sum(np.array(f1s) * confusion.sum(1)) / max(total, 1)),
        "balanced_accuracy": float(np.mean(recalls)),
        "class_f1": {classes[c]: float(f1s[c]) for c in range(n)},
        "class_recall": {classes[c]: float(recalls[c]) for c in range(n)},
        "class_precision": {classes[c]: float(precisions[c]) for c in range(n)},
        "support": {classes[c]: int(confusion[c].sum()) for c in range(n)},
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def predict_logits(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels). Supports multi-view TTA loaders."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        if tta:
            views, labels = batch           # [B, V, 3, H, W]
            b, v = views.shape[0], views.shape[1]
            flat = views.reshape(b * v, *views.shape[2:]).to(device, non_blocking=True)
            logits = model(flat)["main_logits"].float()
            logits = logits.view(b, v, -1).mean(1)
        else:
            images, labels = batch
            logits = model(images.to(device, non_blocking=True))["main_logits"].float()
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, classes: List[str],
                        bias: Optional[np.ndarray] = None) -> Dict:
    if bias is not None:
        logits = logits + bias[None, :]
    pred = logits.argmax(1)
    n = len(classes)
    confusion = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(labels, pred):
        confusion[t, p] += 1
    return confusion_to_metrics(confusion, classes)


def tune_logit_bias(logits: np.ndarray, labels: np.ndarray, classes: List[str],
                    steps: int = 40, lr: float = 0.5) -> np.ndarray:
    """Coordinate ascent on a per-class additive bias to maximise macro-F1.

    Only the validation split is used. This is the standard "logit adjustment"
    fix for imbalanced classification; it changes no weights and no inference
    cost, so it is safe to apply to the deployed student.
    """
    n = len(classes)
    bias = np.zeros(n, dtype=np.float64)
    best = metrics_from_logits(logits, labels, classes, bias)["macro_f1"]
    for _ in range(steps):
        improved = False
        for c in range(n):
            for delta in (-lr, lr):
                trial = bias.copy()
                trial[c] += delta
                score = metrics_from_logits(logits, labels, classes, trial)["macro_f1"]
                if score > best + 1e-6:
                    best, bias, improved = score, trial, True
        if not improved:
            lr *= 0.5
            if lr < 0.02:
                break
    return bias.astype(np.float32)
