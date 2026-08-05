"""Feature engineering and transformation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
TARGET_COLUMN = "is_attack"
LABEL_COLUMN = "label"


def split_features_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    drop_columns = [TARGET_COLUMN, LABEL_COLUMN]

    X_train = train_df.drop(columns=drop_columns, errors="ignore")
    X_test = test_df.drop(columns=drop_columns, errors="ignore")

    y_train = train_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)

    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [column for column in X.columns if column not in CATEGORICAL_COLUMNS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
    )

    return preprocessor


def create_feature_bundle(train_df: pd.DataFrame, test_df: pd.DataFrame) -> FeatureBundle:
    X_train, X_test, y_train, y_test = split_features_target(train_df, test_df)
    preprocessor = build_preprocessor(X_train)

    return FeatureBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
    )
