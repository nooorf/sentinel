"""SYNTHETIC data generator with the Kaggle credit-card schema.

FOR UNIT TESTS, CI SMOKE TESTS AND LOAD TESTS ONLY. Results obtained on this data are NOT benchmark results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.schema import FEATURES, LABEL, TIME_COL


def make_synthetic_creditcard(n: int = 6000, fraud_rate: float = 0.04, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 28))
    # learnable fraud signal in a few components (fraud rows shifted)
    X[y == 1, 0] -= 2.5
    X[y == 1, 1] += 2.0
    X[y == 1, 2] -= 1.5
    X[y == 1, 3] += 1.2
    amount = np.exp(rng.normal(3.5, 1.0, size=n)) * np.where(y == 1, 1.5, 1.0)
    df = pd.DataFrame(X, columns=FEATURES[:-1])
    df["Amount"] = amount.round(2)
    df[TIME_COL] = np.sort(rng.uniform(0, 172800, size=n)).round(0)
    df[LABEL] = y
    return df[[TIME_COL, *FEATURES, LABEL]]
