"""Main entrypoint for training and evaluating the AI-powered NIDS."""

from __future__ import annotations

from pathlib import Path

from src.train_model import train_all_models


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    metrics, comparison_df = train_all_models(data_dir=data_dir, models_dir=models_dir)

    print("Training complete. Metrics summary:")
    for model_name, model_metrics in metrics.items():
        print(f"- {model_name}: {model_metrics}")

    print(f"Best model: {comparison_df.index[0]}")
    print(f"Artifacts stored in: {models_dir}")


if __name__ == "__main__":
    main()
