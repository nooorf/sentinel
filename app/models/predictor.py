"""Inference wrapper. Imports no web/monitoring libraries so it can run inside Azure ML scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.data.schema import FEATURES
from app.models.artifacts import load_artifact
from app.models.scoring import score_parts


class InputValidationError(ValueError):
    pass


class Predictor:
    def __init__(self, bundle: dict, manifest: dict, path: Path):
        self.bundle, self.manifest, self.path = bundle, manifest, Path(path)
        self.version: str = manifest["version"]
        self.threshold: float = float(manifest["threshold"])
        self.decision_model: str = manifest["decision_model"]

    @classmethod
    def load(cls, path: Path) -> Predictor:
        bundle, manifest = load_artifact(path)
        return cls(bundle, manifest, path)

    def info(self) -> dict:
        m = self.manifest
        return {k: m[k] for k in ("version", "created_at", "decision_model", "threshold", "ensemble_weight",
                                  "features", "dataset", "headline_metrics", "libraries", "training_config")}

    def risk(self, score: float) -> str:
        if score >= self.threshold:
            return "HIGH"
        return "MEDIUM" if score >= 0.5 * self.threshold else "LOW"

    def score_frame(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Validate (never reorder-by-guessing or invent values) and score."""
        missing = [c for c in FEATURES if c not in df.columns]
        extra = [c for c in df.columns if c not in FEATURES]
        if missing or extra:
            raise InputValidationError(f"missing features: {missing}; unexpected features: {extra}")
        X = df[FEATURES].astype(float)
        if not np.isfinite(X.to_numpy()).all():
            raise InputValidationError("features must be finite numbers")
        return score_parts(self.bundle, X)

    def predict_frame(self, df: pd.DataFrame) -> list[dict]:
        s = self.score_frame(df)
        out = []
        for p, a, d in zip(s["xgboost"], s["iforest"], s["decision"], strict=True):
            out.append({
                "fraud_probability": float(p), "anomaly_score": float(a), "decision_score": float(d),
                "risk_category": self.risk(float(d)),
                "decision": "REVIEW" if d >= self.threshold else "APPROVE",
                "decision_model": self.decision_model, "model_version": self.version})
        return out
