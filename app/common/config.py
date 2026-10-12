"""Central, environment-driven configuration (all variables prefixed SENTINEL_)."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(f"SENTINEL_{name}", default)


def _bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_path: Path = Path("data/raw/creditcard.csv")
    data_dir: Path = Path("var")
    sample_frac: float | None = None
    split_train: float = 0.6
    split_val: float = 0.2
    seed: int = 42
    target_recall: float = 0.80
    # monitoring
    monitor_window: int = 1000
    monitor_min_samples: int = 200
    drift_share_threshold: float = 0.30
    prediction_drift_triggers: bool = False
    auto_retrain_on_drift: bool = False
    min_labeled_for_performance: int = 30
    # retraining policy
    retrain_min_fraud: int = 20
    retrain_min_new_fraud_labels: int = 0  # drift-triggered runs wait for this many fraud labels on the active model's traffic
    gate_min_recall: float = 0.70
    gate_min_precision: float = 0.30
    gate_min_pr_auc_delta: float = 0.0
    gate_min_val_frauds: int = 30  # below this the validation comparison is too noisy to justify a promotion

    @classmethod
    def from_env(cls) -> Settings:
        frac = _env("SAMPLE_FRAC", "").strip()
        return cls(
            data_path=Path(_env("DATA_PATH", "data/raw/creditcard.csv")),
            data_dir=Path(_env("DATA_DIR", "var")),
            sample_frac=float(frac) if frac else None,
            split_train=float(_env("SPLIT_TRAIN", "0.6")),
            split_val=float(_env("SPLIT_VAL", "0.2")),
            seed=int(_env("SEED", "42")),
            target_recall=float(_env("TARGET_RECALL", "0.80")),
            monitor_window=int(_env("MONITOR_WINDOW", "1000")),
            monitor_min_samples=int(_env("MONITOR_MIN_SAMPLES", "200")),
            drift_share_threshold=float(_env("DRIFT_SHARE_THRESHOLD", "0.30")),
            prediction_drift_triggers=_bool(_env("PREDICTION_DRIFT_TRIGGERS", "false")),
            auto_retrain_on_drift=_bool(_env("AUTO_RETRAIN_ON_DRIFT", "false")),
            retrain_min_fraud=int(_env("RETRAIN_MIN_FRAUD", "20")),
            retrain_min_new_fraud_labels=int(_env("RETRAIN_MIN_NEW_FRAUD_LABELS", "0")),
            gate_min_recall=float(_env("GATE_MIN_RECALL", "0.70")),
            gate_min_precision=float(_env("GATE_MIN_PRECISION", "0.30")),
            gate_min_pr_auc_delta=float(_env("GATE_MIN_PR_AUC_DELTA", "0.0")),
            gate_min_val_frauds=int(_env("GATE_MIN_VAL_FRAUDS", "30")),
        )

    def with_overrides(self, **kw) -> Settings:
        return replace(self, **kw)

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "sentinel.db"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def mlflow_uri(self) -> str:
        return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(self.data_dir / 'mlflow.db').resolve()}")

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.models_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
