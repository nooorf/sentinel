"""Evidently-based drift evaluation.

Three DIFFERENT questions are kept separate:
  1. Feature drift      - did the input distribution move?                  (Evidently DataDriftPreset)
  2. Prediction drift   - did the model's score distribution move?          (Evidently ColumnDriftMetric)
  3. Performance change - did precision/recall change? Needs ground truth; only computed on events that carry
                          a label. Drift in (1)/(2) does NOT prove degradation in (3).

Thresholds: per-column tests use Evidently defaults (KS p-value<0.05 for n<=1000, Wasserstein otherwise).
A dataset is 'drifted' when the share of drifted columns >= drift_share_threshold (default 0.30, configurable).
With 29 features and a 0.05 test level, ~1-2 false-positive columns are expected by chance (3-7%), so 0.30
leaves a wide margin; tune it on historical windows if you have them.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report

from app.data.schema import FEATURES


@dataclass(frozen=True)
class DriftConfig:
    window: int = 1000
    min_samples: int = 200
    share_threshold: float = 0.30
    prediction_drift_triggers: bool = False


def run_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: DriftConfig, html_dir=None, rid="report") -> dict:
    ref_f, cur_f = reference[FEATURES], current[FEATURES]
    rep = Report(metrics=[DataDriftPreset(columns=FEATURES, drift_share=cfg.share_threshold)])
    rep.run(reference_data=ref_f, current_data=cur_f)
    d = rep.as_dict()
    table = next(m["result"] for m in d["metrics"] if "drift_by_columns" in m["result"])
    ds = next(m["result"] for m in d["metrics"] if "dataset_drift" in m["result"] and "drift_share" in m["result"])
    cols = {c: {"drift_detected": bool(v["drift_detected"]), "drift_score": float(v["drift_score"]),
                "stattest": v["stattest_name"]} for c, v in table["drift_by_columns"].items()}

    prep = Report(metrics=[ColumnDriftMetric(column_name="score")])
    prep.run(reference_data=reference[["score"]], current_data=current[["score"]])
    pr = prep.as_dict()["metrics"][0]["result"]

    paths = {}
    if html_dir is not None:
        from pathlib import Path
        Path(html_dir).mkdir(parents=True, exist_ok=True)
        paths = {"feature_html": str(Path(html_dir) / f"{rid}_features.html"),
                 "prediction_html": str(Path(html_dir) / f"{rid}_prediction.html")}
        rep.save_html(paths["feature_html"])
        prep.save_html(paths["prediction_html"])

    return {
        "feature_drift": {"n_features": int(ds["number_of_columns"]), "n_drifted": int(ds["number_of_drifted_columns"]),
                          "share_drifted": float(ds["share_of_drifted_columns"]),
                          "dataset_drift": bool(ds["dataset_drift"]), "columns": cols},
        "prediction_drift": {"drift_detected": bool(pr["drift_detected"]), "drift_score": float(pr["drift_score"]),
                             "stattest": pr["stattest_name"]},
        "paths": paths,
    }


def performance_summary(events: pd.DataFrame, min_labeled: int) -> dict:
    """Performance can only be assessed on labeled events. Never invents labels."""
    lab = events[events["label"].notna()] if "label" in events else events.iloc[0:0]
    if len(lab) < min_labeled or lab["label"].nunique() < 2:
        return {"status": "unavailable", "labeled_events": int(len(lab)),
                "note": f"Needs >= {min_labeled} labeled events including both classes; ground truth arrives with delay."}
    y, f = lab["label"].astype(int), lab["flagged"].astype(int)
    tp, fp, fn = int(((y == 1) & (f == 1)).sum()), int(((y == 0) & (f == 1)).sum()), int(((y == 1) & (f == 0)).sum())
    return {"status": "computed", "labeled_events": int(len(lab)), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}


def drift_condition(summary: dict, cfg: DriftConfig) -> tuple[bool, str]:
    """Pure function: should this monitoring result trigger retraining?"""
    if summary.get("status") != "ok":
        return False, f"monitoring status is '{summary.get('status')}' (not enough data or failed)"
    fd = summary["feature_drift"]
    if fd["share_drifted"] >= cfg.share_threshold:
        return True, f"feature drift: {fd['n_drifted']}/{fd['n_features']} features drifted (share {fd['share_drifted']:.2f} >= {cfg.share_threshold})"
    if cfg.prediction_drift_triggers and summary["prediction_drift"]["drift_detected"]:
        return True, "prediction-distribution drift detected (prediction_drift_triggers enabled)"
    return False, "drift below configured thresholds"
