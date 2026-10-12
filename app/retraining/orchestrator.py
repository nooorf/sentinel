"""Retraining orchestrator: run state machine, duplicate-run guard, candidate gate, promotion, rollback.

States: pending -> checking -> training -> evaluating -> promoted | rejected | failed | skipped
Training always uses the HISTORICAL LABELED dataset followed by any real, labeled monitoring events that arrived with
delayed ground truth (appended in arrival order, so the newest labeled traffic lands in validation/test).
Unlabeled and simulated events are never used as training data and labels are never invented.
Execution runs on a single background worker thread; prediction requests are never blocked by training.
"""
from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from app.common.config import Settings
from app.common.jsonutil import utcnow
from app.common.logging import get_logger, log
from app.data.schema import LABEL, Splits
from app.models.artifacts import load_artifact
from app.models.metrics import compute_metrics
from app.models.pipeline import prepare_splits, train_and_save
from app.models.scoring import score_parts
from app.monitoring.store import Store
from app.registry.registry import ModelRegistry

logger = get_logger()
TERMINAL = {"promoted", "rejected", "failed", "skipped"}


class RetrainingConflict(RuntimeError):
    pass


class RetrainingRejected(ValueError):
    pass


def evaluate_gate(cand: dict, active: dict | None, s: Settings) -> tuple[bool, list[str]]:
    """Promotion policy. Both metric dicts come from the SAME validation split. Returns (accept, reasons)."""
    reasons = []
    if cand["n_fraud"] < s.gate_min_val_frauds:  # e.g. 10 frauds: recall moves in 10% steps, PR-AUC is mostly noise
        reasons.append(f"insufficient evidence: {cand['n_fraud']} frauds in validation (< {s.gate_min_val_frauds})")
    if cand["recall"] < s.gate_min_recall:
        reasons.append(f"recall {cand['recall']:.3f} < required {s.gate_min_recall}")
    if cand["precision"] < s.gate_min_precision:
        reasons.append(f"precision {cand['precision']:.3f} < required {s.gate_min_precision}")
    if active is not None and cand["pr_auc"] < active["pr_auc"] + s.gate_min_pr_auc_delta:
        reasons.append(f"PR-AUC {cand['pr_auc']:.4f} < active {active['pr_auc']:.4f} + delta {s.gate_min_pr_auc_delta}")
    return (not reasons), reasons


class Orchestrator:
    def __init__(self, settings: Settings, store: Store, registry: ModelRegistry,
                 on_model_change: Callable[[str], None] | None = None):
        self.s, self.store, self.registry = settings, store, registry
        self.on_model_change = on_model_change or (lambda v: None)
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrain")
        self.mark_stale_runs()

    def mark_stale_runs(self) -> None:
        """Runs left active by a crashed process can never finish; mark them failed so new runs are possible."""
        for r in self.store.active_runs():
            tr = (r["transitions"] or []) + [{"ts": utcnow(), "status": "failed", "note": "interrupted by restart"}]
            self.store.update_run(r["id"], status="failed", finished_at=utcnow(), error="interrupted by service restart",
                                  transitions=tr)

    # -------------------------------------------------------------- trigger
    def trigger(self, trigger: str = "manual", reason: str | None = None, overrides: dict | None = None,
                drift_report_id: str | None = None, background: bool = True) -> dict:
        with self._lock:
            if self.store.active_runs():
                raise RetrainingConflict("a retraining run is already active")
            if trigger == "drift":
                rep = self.store.get_report(drift_report_id) if drift_report_id else None
                if rep is None or rep.get("status") != "ok":
                    raise RetrainingRejected("drift-triggered retraining needs a completed monitoring report")
                if rep["current"]["rows"] < self.s.monitor_min_samples:
                    raise RetrainingRejected("monitoring window below minimum sample size")
                if not rep.get("drift_detected"):
                    raise RetrainingRejected("drift condition not met for this report")
                need = self.s.retrain_min_new_fraud_labels
                if need:
                    have = self.store.count_labeled(self.registry.active())
                    if have < need:
                        raise RetrainingRejected(f"waiting for ground truth: {have} fraud labels on the active model's "
                                                 f"traffic (< {need}); drift alone does not justify a new model")
                reason = reason or rep.get("drift_reason")
            from app.models.train import TrainConfig
            TrainConfig().with_overrides(overrides)  # validate overrides early (raises ValueError)
            rid = f"run-{uuid.uuid4().hex[:10]}"
            self.store.create_run({
                "id": rid, "trigger": trigger, "status": "pending", "requested_at": utcnow(), "reason": reason,
                "drift_report_id": drift_report_id, "overrides": overrides, "active_version": self.registry.active(),
                "transitions": [{"ts": utcnow(), "status": "pending", "note": f"trigger={trigger}"}]})
        log(logger, "retraining_requested", run_id=rid, trigger=trigger)
        if background:
            self._pool.submit(self._execute, rid)
        else:
            self._execute(rid)
        return self.store.get_run(rid)

    # -------------------------------------------------------------- state machine
    def _set(self, rid: str, status: str, note: str = "", **fields) -> None:
        run = self.store.get_run(rid)
        tr = (run["transitions"] or []) + [{"ts": utcnow(), "status": status, "note": note}]
        extra = {"finished_at": utcnow()} if status in TERMINAL else {}
        if status == "checking":
            extra["started_at"] = utcnow()
        self.store.update_run(rid, status=status, transitions=tr, **extra, **fields)
        log(logger, "retraining_transition", run_id=rid, status=status, note=note)

    def _execute(self, rid: str) -> None:
        try:
            run = self.store.get_run(rid)
            self._set(rid, "checking")
            splits, dsinfo = prepare_splits(self.s, self.store.labeled_events())
            n_fraud = int(splits.train[LABEL].sum())
            if n_fraud < self.s.retrain_min_fraud:
                self._set(rid, "skipped", decision=f"only {n_fraud} fraud rows in training data (< {self.s.retrain_min_fraud})")
                return

            self._set(rid, "training")
            version, _, bundle, splits = train_and_save(
                self.s, run["overrides"], splits_info=(splits, dsinfo), run_name=rid,
                tags={"retraining_run": rid, "trigger": run["trigger"]})
            cand_val = bundle["report"]["models"][bundle["decision_model"]]["validation"]
            self.registry.register_candidate(version, {"validation": cand_val}, reason=f"candidate from {rid}")

            self._set(rid, "evaluating", note=f"candidate={version}", candidate_version=version)
            active_val = self._evaluate_active(splits)
            ok, reasons = evaluate_gate(cand_val, active_val, self.s)
            fields = dict(candidate_metrics=cand_val, active_metrics=active_val)
            if ok:
                load_artifact(self.registry.artifact_path(version))  # integrity check BEFORE changing the active pointer
                decision = "promoted: enough validation frauds, passed recall/precision floors, PR-AUC not below active model"
                self.registry.promote(version, decision)
                self.on_model_change(version)
                self._set(rid, "promoted", decision=decision, **fields)
            else:
                decision = "rejected: " + "; ".join(reasons)
                self.registry.reject(version, decision)
                self._set(rid, "rejected", decision=decision, **fields)  # active model untouched
        except Exception as e:  # noqa: BLE001
            log(logger, "retraining_failed", run_id=rid, error=str(e))
            self._set(rid, "failed", note=type(e).__name__, error=f"{type(e).__name__}: {e}")

    def _evaluate_active(self, splits: Splits) -> dict | None:
        """Active model re-scored on the SAME validation split used for the candidate."""
        active = self.registry.active()
        if active is None:
            return None
        bundle, manifest = load_artifact(self.registry.artifact_path(active))
        sv = score_parts(bundle, splits.val)["decision"]
        return compute_metrics(splits.val[LABEL].to_numpy(), sv, manifest["threshold"])

    # -------------------------------------------------------------- rollback
    def rollback(self, reason: str, to: str | None = None) -> str:
        with self._lock:
            if self.store.active_runs():
                raise RetrainingConflict("cannot roll back while a retraining run is active")
            snap = self.registry.snapshot()
            stack = snap["approved_stack"]
            pre = to or (stack[-2] if len(stack) >= 2 else None)
            if pre:  # verify the target loads before touching the active pointer
                load_artifact(self.registry.artifact_path(pre))
            target = self.registry.rollback(reason, to)
            self.on_model_change(target)
            log(logger, "model_rolled_back", to=target)
            return target

    def wait_idle(self) -> None:
        self._pool.submit(lambda: None).result()
