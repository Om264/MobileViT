"""Evaluation loop, classification reports and plots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .utils import autocast_ctx, prep


@torch.no_grad()
def evaluate(model, loader, device, amp_on=False, amp_dtype=torch.float16, channels_last=False):
    """Run the model over `loader`. Returns loss/acc/macro_f1/weighted_f1 plus y_true / y_pred arrays."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total, ys, ps = 0.0, [], []
    for x, y in loader:
        x = prep(x, device, channels_last)
        y = y.to(device)
        with autocast_ctx(device, amp_on, amp_dtype):
            out = model(x)
        total += crit(out.float(), y).item()
        ps.append(out.argmax(1).cpu())
        ys.append(y.cpu())
    y_true, y_pred = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {
        "loss": total / len(y_true),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_confusion(cm, class_names, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(class_names)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(6, n * 0.3)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    fs = 8 if n <= 20 else 5
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=fs)
    ax.set_yticklabels(class_names, fontsize=fs)
    if n <= 20:
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_reports(y_true, y_pred, class_names, out_dir, tag):
    """Write <tag>_report.txt/.json and <tag>_confusion_matrix.csv/.png. Returns the text report."""
    out_dir = Path(out_dir)
    labels = list(range(len(class_names)))
    kw = dict(labels=labels, target_names=class_names, zero_division=0)
    txt = classification_report(y_true, y_pred, digits=4, **kw)
    (out_dir / f"{tag}_report.txt").write_text(txt)
    (out_dir / f"{tag}_report.json").write_text(
        json.dumps(classification_report(y_true, y_pred, output_dict=True, **kw), indent=2)
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(out_dir / f"{tag}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, class_names, out_dir / f"{tag}_confusion_matrix.png")
    return txt


def plot_history(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val acc")
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], label="val macro-F1")
    ax[1].set_title("Validation metrics")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
