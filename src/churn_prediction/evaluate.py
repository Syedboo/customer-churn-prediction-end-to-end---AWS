from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Retained", "Churn"], ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_pr_curves(y_true, probabilities, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=axes[1])
    axes[0].set_title("ROC Curve")
    axes[1].set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
