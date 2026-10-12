"""Monitoring orchestration: take a recent window, compare with the active model's reference, persist a report."""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from app.common.config import Settings
from app.common.jsonutil import utcnow
from app.models.predictor import Predictor
from app.monitoring.drift import DriftConfig, drift_condition, performance_summary, run_drift
from app.monitoring.store import Store


class MonitoringService:
    def __init__(self, settings: Settings, store: Store):
        self.s, self.store = settings, store
        self._ref_cache: dict[str, pd.DataFrame] = {}

    def config(self) -> DriftConfig:
        return DriftConfig(self.s.monitor_window, self.s.monitor_min_samples, self.s.drift_share_threshold,
                           self.s.prediction_drift_triggers)

    def reference(self, predictor: Predictor) -> pd.DataFrame:
        if predictor.version not in self._ref_cache:
            self._ref_cache = {predictor.version: pd.read_csv(Path(predictor.path) / "reference.csv.gz")}
        return self._ref_cache[predictor.version]

    def evaluate(self, predictor: Predictor, window: int | None = None, source: str | None = None,
                 batch_tag: str | None = None, simulated: bool = False) -> dict:
        cfg = self.config()
        window = window or cfg.window
        rid = f"mon-{uuid.uuid4().hex[:10]}"
        events = self.store.recent_events(window, source=source, batch_tag=batch_tag, model_version=predictor.version)
        ref = self.reference(predictor)
        base = {"id": rid, "created_at": utcnow(), "model_version": predictor.version, "source": source,
                "batch_tag": batch_tag, "contains_simulated_events": bool(
                    simulated or (len(events) and (events["source"] == "simulated").any())),
                "config": {"window": window, "min_samples": cfg.min_samples, "share_threshold": cfg.share_threshold,
                           "prediction_drift_triggers": cfg.prediction_drift_triggers},
                "reference": {"rows": int(len(ref)), "origin": "validation split sample of the active model's training data"},
                "current": {"rows": int(len(events))}}
        if len(events) < cfg.min_samples:
            summary = {**base, "status": "insufficient_data", "drift_detected": None,
                       "message": f"{len(events)} events in window; need >= {cfg.min_samples}."}
        else:
            try:
                res = run_drift(ref, events, cfg, html_dir=self.s.reports_dir / "drift", rid=rid)
                summary = {**base, "status": "ok", **res,
                           "performance": performance_summary(events, self.s.min_labeled_for_performance)}
                trig, why = drift_condition(summary, cfg)
                summary.update(drift_detected=trig, drift_reason=why)
            except Exception as e:  # noqa: BLE001
                summary = {**base, "status": "failed", "drift_detected": None, "message": f"{type(e).__name__}: {e}"}
        self.store.save_report(rid, summary)
        return summary

    def latest(self) -> dict | None:
        return self.store.latest_report()
