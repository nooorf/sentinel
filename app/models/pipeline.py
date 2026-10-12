"""End-to-end: load data -> split -> train -> persist artifact (+ MLflow tracking)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.common.config import Settings
from app.common.logging import get_logger, log
from app.data.schema import LABEL, TIME_COL, Splits, chronological_split, load_dataset
from app.models.artifacts import save_artifact
from app.models.train import TrainConfig, train_pipeline

logger = get_logger()


def prepare_splits(settings: Settings, labeled_events: pd.DataFrame | None = None) -> tuple[Splits, dict]:
    """Historical data, optionally followed by labelled monitoring events.

    Events happened after the historical data, so they are appended in arrival order and the chronological split
    puts the newest labelled traffic into validation/test: the gate then judges candidate and active model on the
    most recent data."""
    df, info = load_dataset(settings.data_path, settings.sample_frac, settings.seed)
    if labeled_events is not None and len(labeled_events):
        ev = labeled_events.assign(**{TIME_COL: df[TIME_COL].max() + 1 + np.arange(len(labeled_events))})
        df = pd.concat([df, ev[df.columns]], ignore_index=True)
        info.update(labeled_events_appended=int(len(ev)), labeled_event_frauds=int(ev[LABEL].sum()),
                    rows_out=int(len(df)), fraud_rows=int(df[LABEL].sum()), fraud_rate=float(df[LABEL].mean()))
    return chronological_split(df, settings.split_train, settings.split_val), info


def log_to_mlflow(settings: Settings, version: str, bundle: dict, run_name: str, tags: dict | None = None) -> str | None:
    """Best-effort experiment tracking. Failure is logged loudly and returned as None (never faked)."""
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_uri)
        mlflow.set_experiment("sentinel")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({"model_version": version, **(tags or {})})
            mlflow.log_params({k: v for k, v in bundle["config"].items() if not isinstance(v, list)})
            mlflow.log_param("decision_model", bundle["decision_model"])
            for name, m in bundle["report"]["models"].items():
                for split in ("validation", "test"):
                    for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_count"):
                        v = m[split].get(k)
                        if v is not None:
                            mlflow.log_metric(f"{name}_{split}_{k}", float(v))
            return run.info.run_id
    except Exception as e:  # noqa: BLE001
        log(logger, "mlflow_tracking_failed", error=str(e))
        return None


def train_and_save(settings: Settings, overrides: dict | None = None, splits_info=None, run_name="train",
                   tags: dict | None = None):
    """Returns (version, path, bundle, splits). `splits_info` may pass pre-built (splits, dataset_info)."""
    settings.ensure_dirs()
    splits, dsinfo = splits_info or prepare_splits(settings)
    cfg = TrainConfig(seed=settings.seed, target_recall=settings.target_recall).with_overrides(overrides)
    bundle = train_pipeline(splits, cfg)
    version, path = save_artifact(bundle, splits, dsinfo, settings.models_dir, settings.seed)
    run_id = log_to_mlflow(settings, version, bundle, run_name, tags)
    log(logger, "model_trained", version=version, decision_model=bundle["decision_model"], mlflow_run=run_id)
    return version, path, bundle, splits
