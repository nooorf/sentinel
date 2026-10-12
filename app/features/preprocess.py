"""Preprocessing pipeline. Fitted on the TRAINING split only and persisted inside the model artifact.

`Time` is deliberately excluded from model features: it is seconds-since-first-transaction (non-stationary and
meaningless for a new transaction); it is only used to order the chronological split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def log_amount(X: pd.DataFrame) -> pd.DataFrame:  # module-level so it is picklable
    X = X.copy()
    X["Amount"] = np.log1p(X["Amount"].clip(lower=0))
    return X


def build_preprocessor() -> Pipeline:
    return Pipeline([("log_amount", FunctionTransformer(log_amount, validate=False)),
                     ("scale", StandardScaler())])
