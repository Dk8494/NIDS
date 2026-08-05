"""Model evaluation utilities and visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: str | Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_model_comparison(results: Dict[str, Dict[str, float]], output_path: str | Path) -> pd.DataFrame:
    comparison_df = pd.DataFrame(results).T.sort_values(by="f1_score", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison_df.reset_index(), x="index", y="f1_score", palette="viridis")
    plt.title("Model F1 Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return comparison_df
