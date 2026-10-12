"""Controlled, clearly-labelled drift simulation (events are tagged source='simulated').

Baseline batch : unshifted rows from the model's held-out test-split pool.
Shifted batch  : same kind of rows with chosen features shifted by `magnitude` standard deviations and Amount
                 multiplied by `amount_factor`. No labels are attached.
NOTE: the baseline may itself show some drift vs. the validation reference because the pool is from a LATER time
window (this is real temporal drift in the dataset, not an error).
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from app.data.schema import FEATURES
from app.models.predictor import Predictor
from app.monitoring.service import MonitoringService
from app.monitoring.store import Store

DEFAULT_SHIFT = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"]


def make_shifted(pool: pd.DataFrame, n: int, features: list[str], magnitude: float, amount_factor: float,
                 seed: int) -> pd.DataFrame:
    bad = [f for f in features if f not in FEATURES]
    if bad:
        raise ValueError(f"unknown features: {bad}")
    df = pool.sample(n=n, replace=n > len(pool), random_state=seed).reset_index(drop=True)
    sd = pool[features].std()
    for f in features:
        df[f] = df[f] + magnitude * sd[f]
    df["Amount"] = df["Amount"] * amount_factor
    return df


def ingest(store: Store, predictor: Predictor, df: pd.DataFrame, tag: str) -> int:
    preds = predictor.predict_frame(df[FEATURES])
    rows = [{"features": r.to_dict(), "prob": p["fraud_probability"], "anomaly": p["anomaly_score"],
             "score": p["decision_score"], "risk": p["risk_category"], "flagged": p["decision"] == "REVIEW"}
            for (_, r), p in zip(df[FEATURES].iterrows(), preds, strict=True)]
    return store.add_events(rows, "simulated", tag, predictor.version)


def run_simulation(service: MonitoringService, store: Store, predictor: Predictor, n: int = 500,
                   features: list[str] | None = None, magnitude: float = 2.0, amount_factor: float = 3.0,
                   seed: int = 42) -> dict:
    features = features or DEFAULT_SHIFT
    pool = pd.read_csv(Path(predictor.path) / "pool.csv.gz")
    sim_id = f"sim-{uuid.uuid4().hex[:8]}"
    base_df = pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)
    shifted_df = make_shifted(pool, n, features, magnitude, amount_factor, seed + 1)
    ingest(store, predictor, base_df, f"{sim_id}-baseline")
    ingest(store, predictor, shifted_df, f"{sim_id}-shifted")
    before = service.evaluate(predictor, window=n, batch_tag=f"{sim_id}-baseline", simulated=True)
    after = service.evaluate(predictor, window=n, batch_tag=f"{sim_id}-shifted", simulated=True)
    return {"simulation_id": sim_id, "label": "SIMULATED - synthetic distribution shift, not real traffic",
            "parameters": {"n": n, "features": features, "magnitude_std": magnitude, "amount_factor": amount_factor,
                           "seed": seed},
            "before": before, "after": after}
