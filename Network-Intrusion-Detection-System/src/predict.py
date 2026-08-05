"""Prediction utilities for intrusion detection."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_dataframe


class IntrusionPredictor:
    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir = Path(models_dir)
        self.preprocessor = joblib.load(self.models_dir / "preprocessor.joblib")

        with open(self.models_dir / "best_model_metadata.json", "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.best_model_name = self.metadata["best_model"]

        if self.metadata["is_deep_learning"]:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(
                self.models_dir / "deep_neural_network.keras"
            )
            self.model_kind = "deep_learning"
        else:
            model_path = self.models_dir / f"{self.best_model_name}.joblib"
            self.model = joblib.load(model_path)
            self.model_kind = "machine_learning"

    def predict_dataframe(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if "is_attack" not in input_df.columns and "label" in input_df.columns:
            input_df = preprocess_dataframe(input_df)

        feature_df = input_df.drop(columns=["is_attack", "label", "difficulty"], errors="ignore")
        transformed = self.preprocessor.transform(feature_df)

        if self.model_kind == "deep_learning":
            transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
            probabilities = self.model.predict(np.asarray(transformed), verbose=0).flatten()
            predictions = (probabilities >= 0.5).astype(int)
        else:
            probabilities = self.model.predict_proba(transformed)[:, 1]
            predictions = self.model.predict(transformed)

        output = input_df.copy()
        output["intrusion_prediction"] = predictions
        output["intrusion_probability"] = probabilities
        output["prediction_label"] = output["intrusion_prediction"].map({0: "normal", 1: "attack"})

        return output


def predict_from_csv(input_csv: str | Path, output_csv: str | Path, models_dir: str | Path = "models") -> Path:
    predictor = IntrusionPredictor(models_dir=models_dir)
    df = pd.read_csv(input_csv)
    predictions_df = predictor.predict_dataframe(df)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    return output_path
