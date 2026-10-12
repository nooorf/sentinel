"""Shared scoring math used by training, serving, the retraining gate and Azure scoring.

Score definitions
- fraud_probability: XGBoost predicted P(fraud) (supervised, trained on labels).
- anomaly_score: Isolation Forest, trained WITHOUT labels. raw = -score_samples (higher = more anomalous),
  mapped to [0,1] as the percentile rank of raw within the TRAINING-set raw-score distribution
  (0.99 means "more anomalous than 99% of training transactions"). It is NOT a probability.
- decision_score: the score compared with the threshold. Equals fraud_probability when the selected decision
  model is 'xgboost', or w*prob + (1-w)*anomaly when the validated ensemble was selected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.schema import FEATURES


def anomaly_from_raw(raw: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    return np.interp(raw, quantiles, np.linspace(0.0, 1.0, len(quantiles)))


def score_parts(bundle: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Score a frame containing exactly the model features. Returns all component scores."""
    Z = bundle["preprocessor"].transform(X[FEATURES])
    prob = bundle["xgb"].predict_proba(Z)[:, 1]
    raw = -bundle["iforest"].score_samples(Z)
    anomaly = anomaly_from_raw(raw, np.asarray(bundle["if_quantiles"]))
    base = bundle["baseline"].predict_proba(Z)[:, 1]
    w = float(bundle["ensemble_weight"])
    return {"baseline": base, "xgboost": prob, "iforest": anomaly, "ensemble": w * prob + (1 - w) * anomaly,
            "decision": (w * prob + (1 - w) * anomaly) if bundle["decision_model"] == "ensemble" else prob}
