"""Dataset schema, validation, loading and chronological splitting (Kaggle credit-card fraud schema)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TIME_COL = "Time"
LABEL = "Class"
REQUIRED = [TIME_COL, *FEATURES, LABEL]

DOWNLOAD_HELP = (
    "Dataset not found at '{path}'. Download 'creditcard.csv' from "
    "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (free Kaggle account required; "
    "licence: Database Contents License DbCL v1.0) and place it at that path, or set SENTINEL_DATA_PATH."
)


class DataValidationError(ValueError):
    pass


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    info: dict


def validate_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate schema and clean rows. Returns (clean_df, report). Raises on schema errors."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    report: dict = {"rows_in": int(len(df))}
    df = df[REQUIRED].copy()
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    n0 = len(df)
    df = df.dropna()
    report["dropped_missing_or_invalid"] = int(n0 - len(df))
    n0 = len(df)
    df = df[df[LABEL].isin([0, 1])]
    report["dropped_bad_label"] = int(n0 - len(df))
    n0 = len(df)
    df = df[df["Amount"] >= 0]
    report["dropped_negative_amount"] = int(n0 - len(df))
    n0 = len(df)
    df = df.drop_duplicates()
    report["dropped_duplicates"] = int(n0 - len(df))
    df[LABEL] = df[LABEL].astype(int)
    if df.empty or df[LABEL].sum() == 0:
        raise DataValidationError("No usable rows or no fraud cases after validation")
    report.update(rows_out=int(len(df)), fraud_rows=int(df[LABEL].sum()), fraud_rate=float(df[LABEL].mean()))
    return df.reset_index(drop=True), report


def load_dataset(path: Path, sample_frac: float | None = None, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(DOWNLOAD_HELP.format(path=path))
    df, report = validate_raw(pd.read_csv(path))
    report["path"] = str(path)
    report["mode"] = "full"
    if sample_frac:
        if not 0 < sample_frac <= 1:
            raise DataValidationError("sample_frac must be in (0, 1]")
        fraud = df[df[LABEL] == 1]
        legit = df[df[LABEL] == 0].sample(frac=sample_frac, random_state=seed)
        df = pd.concat([fraud, legit]).sort_values(TIME_COL, kind="stable").reset_index(drop=True)
        report.update(mode=f"DEV SAMPLE (all fraud + {sample_frac:.0%} of legit) - NOT the full benchmark",
                      fraud_rows=int(df[LABEL].sum()),
                      rows_out=int(len(df)), fraud_rate=float(df[LABEL].mean()))
    return df, report


def chronological_split(df: pd.DataFrame, train: float = 0.6, val: float = 0.2,
                        min_fraud_per_split: int = 5) -> Splits:
    """Split by transaction time (oldest -> train, newest -> test). Raises if a split has too few frauds."""
    if train + val >= 1:
        raise DataValidationError("train + val fractions must be < 1")
    d = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    i1, i2 = int(len(d) * train), int(len(d) * (train + val))
    parts = {"train": d.iloc[:i1], "val": d.iloc[i1:i2], "test": d.iloc[i2:]}
    info = {}
    for name, p in parts.items():
        nf = int(p[LABEL].sum())
        info[name] = {"rows": int(len(p)), "fraud": nf, "fraud_rate": float(p[LABEL].mean()) if len(p) else 0.0,
                      "time_min": float(p[TIME_COL].min()), "time_max": float(p[TIME_COL].max())}
        if nf < min_fraud_per_split:
            raise DataValidationError(
                f"Split '{name}' has only {nf} fraud cases (< {min_fraud_per_split}); "
                "use more data (larger sample_frac) or a different split.")
    return Splits(parts["train"].reset_index(drop=True), parts["val"].reset_index(drop=True),
                  parts["test"].reset_index(drop=True), info)
