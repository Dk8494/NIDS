"""Train ML and DL intrusion detection models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluate_model import compute_metrics, plot_confusion_matrix, save_model_comparison
from src.feature_engineering import create_feature_bundle
from src.preprocessing import get_train_test_data


def _to_dense(array_like):
    return array_like.toarray() if hasattr(array_like, "toarray") else array_like


def _build_dnn(input_dim: int):
    import tensorflow as tf
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _train_sklearn_models(X_train, y_train, random_state: int = 42):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            n_jobs=1,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        ),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def train_all_models(
    data_dir: str | Path = "data",
    models_dir: str | Path = "models",
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    train_df, test_df = get_train_test_data(data_dir)
    bundle = create_feature_bundle(train_df, test_df)

    X_train_transformed = bundle.preprocessor.fit_transform(bundle.X_train)
    X_test_transformed = bundle.preprocessor.transform(bundle.X_test)

    joblib.dump(bundle.preprocessor, models_path / "preprocessor.joblib")

    sklearn_models = _train_sklearn_models(X_train_transformed, bundle.y_train, random_state=random_state)

    results: Dict[str, Dict[str, float]] = {}

    for model_name, model in sklearn_models.items():
        predictions = model.predict(X_test_transformed)
        metrics = compute_metrics(bundle.y_test, predictions)
        results[model_name] = metrics

        joblib.dump(model, models_path / f"{model_name}.joblib")
        plot_confusion_matrix(
            bundle.y_test,
            predictions,
            title=f"{model_name} Confusion Matrix",
            output_path=models_path / f"{model_name}_confusion_matrix.png",
        )

    X_train_dense = _to_dense(X_train_transformed).astype(np.float32)
    X_test_dense = _to_dense(X_test_transformed).astype(np.float32)

    import tensorflow as tf
    dnn_model = _build_dnn(input_dim=X_train_dense.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    dnn_model.fit(
        X_train_dense,
        bundle.y_train,
        validation_split=0.2,
        epochs=25,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=0,
    )

    dnn_predictions = (dnn_model.predict(X_test_dense, verbose=0).flatten() >= 0.5).astype(int)
    dnn_metrics = compute_metrics(bundle.y_test, dnn_predictions)
    results["deep_neural_network"] = dnn_metrics

    dnn_model.save(models_path / "deep_neural_network.keras")
    plot_confusion_matrix(
        bundle.y_test,
        dnn_predictions,
        title="Deep Neural Network Confusion Matrix",
        output_path=models_path / "deep_neural_network_confusion_matrix.png",
    )

    comparison_df = save_model_comparison(results, models_path / "model_comparison_f1.png")
    comparison_df.to_csv(models_path / "metrics_summary.csv", index=True)

    best_model_name = comparison_df.index[0]
    best_model_meta = {
        "best_model": best_model_name,
        "is_deep_learning": best_model_name == "deep_neural_network",
        "all_metrics": results,
    }

    with open(models_path / "best_model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(best_model_meta, file, indent=2)

    if best_model_name != "deep_neural_network":
        joblib.dump(sklearn_models[best_model_name], models_path / "best_model.joblib")

    return results, comparison_df


if __name__ == "__main__":
    metrics, comparison = train_all_models()
    print("Training completed. Model metrics:")
    print(pd.DataFrame(metrics).T)
    print("\nBest model:", comparison.index[0])
