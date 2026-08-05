"""Data ingestion and preprocessing utilities for NSL-KDD."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import requests


COLUMN_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

TRAIN_URLS = [
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
]

TEST_URLS = [
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
]


def _download_with_fallback(urls: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    last_error = None
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            destination.write_bytes(response.content)
            return
        except requests.RequestException as exc:
            last_error = exc

    raise RuntimeError(f"Failed to download dataset file: {destination.name}") from last_error


def download_nsl_kdd_if_missing(data_dir: str | Path = "data") -> Tuple[Path, Path]:
    data_path = Path(data_dir)
    train_file = data_path / "KDDTrain+.txt"
    test_file = data_path / "KDDTest+.txt"

    if not train_file.exists():
        _download_with_fallback(TRAIN_URLS, train_file)

    if not test_file.exists():
        _download_with_fallback(TEST_URLS, test_file)

    return train_file, test_file


def load_nsl_kdd(data_dir: str | Path = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_file, test_file = download_nsl_kdd_if_missing(data_dir)

    train_df = pd.read_csv(train_file, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_file, names=COLUMN_NAMES)

    return train_df, test_df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    working_df = working_df.drop(columns=["difficulty"], errors="ignore")
    working_df["is_attack"] = (working_df["label"] != "normal").astype(int)

    return working_df


def get_train_test_data(data_dir: str | Path = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = load_nsl_kdd(data_dir)
    return preprocess_dataframe(train_df), preprocess_dataframe(test_df)
