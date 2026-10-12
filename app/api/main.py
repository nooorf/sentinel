"""Sentinel inference + monitoring + retraining API."""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.schemas import (
    BatchRequest,
    EvaluateRequest,
    EventsRequest,
    Prediction,
    RetrainRequest,
    RollbackRequest,
    SimulateRequest,
    Transaction,
)
from app.common.config import Settings
from app.common.logging import get_logger, log
from app.models.artifacts import ArtifactError
from app.models.predictor import InputValidationError, Predictor
from app.monitoring.service import MonitoringService
from app.monitoring.simulate import run_simulation
from app.monitoring.store import Store
from app.registry.registry import ModelRegistry, RegistryError
from app.retraining.orchestrator import Orchestrator, RetrainingConflict, RetrainingRejected

logger = get_logger()
STATIC = Path(__file__).resolve().parents[1] / "dashboard" / "static"


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.ensure_dirs()
        self.registry = ModelRegistry(settings.models_dir)
        self.store = Store(settings.db_path)
        self.monitor = MonitoringService(settings, self.store)
        self.orchestrator = Orchestrator(settings, self.store, self.registry, self.load_model)
        self.predictor: Predictor | None = None
        self._lock = threading.Lock()
        self.started = time.time()
        self.counts: Counter = Counter()
        self.latencies: deque = deque(maxlen=5000)
        self.load_error: str | None = None

    def load_model(self, version: str | None = None) -> None:
        version = version or self.registry.active()
        if version is None:
            return
        p = Predictor.load(self.registry.artifact_path(version))  # verify before swapping
        with self._lock:
            self.predictor = p

    def require_model(self) -> Predictor:
        if self.predictor is None:
            raise HTTPException(503, detail="No model loaded. Train one first: python scripts/train.py")
        return self.predictor


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st = AppState(settings)
        try:
            st.load_model()
        except (ArtifactError, FileNotFoundError, RegistryError) as e:
            st.load_error = str(e)
            log(logger, "model_load_failed", error=str(e))
        app.state.s = st
        log(logger, "startup", model=st.predictor.version if st.predictor else None)
        yield

    app = FastAPI(title="Sentinel Fraud Detection API", version="0.1.0", lifespan=lifespan,
                  description="Portfolio demonstration; trained on a public dataset. Not a production payment processor.")

    @app.middleware("http")
    async def metrics_mw(request: Request, call_next):
        t = time.perf_counter()
        status = 500
        try:
            resp = await call_next(request)
            status = resp.status_code
            return resp
        finally:
            ms = (time.perf_counter() - t) * 1000
            st: AppState | None = getattr(request.app.state, "s", None)
            if st:
                path = request.scope.get("route").path if request.scope.get("route") else "unmatched"
                st.counts[f"{request.method} {path}"] += 1
                if status >= 400:
                    st.counts["errors_total"] += 1
                    st.counts[f"errors_{status}"] += 1
                st.counts["requests_total"] += 1
                if path in ("/predict", "/predict/batch"):
                    st.latencies.append(ms)
            log(logger, "request", method=request.method, path=request.url.path, status=status, ms=round(ms, 2))

    def S(request: Request) -> AppState:
        return request.app.state.s

    # ------------------------------------------------------------------ errors
    @app.exception_handler(InputValidationError)
    async def _iv(_, e):
        return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": str(e)})

    # ------------------------------------------------------------------ dashboard (static, talks to this API over HTTP)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC / "index.html")

    # ------------------------------------------------------------------ health / model
    @app.get("/health")
    def health(request: Request):
        st = S(request)
        return {"status": "ok" if st.predictor else "degraded", "model_loaded": st.predictor is not None,
                "model_version": st.predictor.version if st.predictor else None, "load_error": st.load_error,
                "uptime_s": round(time.time() - st.started, 1)}

    @app.get("/model/info")
    def model_info(request: Request):
        return S(request).require_model().info()

    @app.get("/model/evaluation")
    def model_eval(request: Request):
        import json
        return json.loads((S(request).require_model().path / "metrics.json").read_text())

    @app.get("/model/examples")
    def model_examples(request: Request):
        import json
        return json.loads((S(request).require_model().path / "examples.json").read_text())

    @app.get("/model/versions")
    def model_versions(request: Request):
        return S(request).registry.snapshot()

    # ------------------------------------------------------------------ prediction
    def _log_events(st: AppState, p: Predictor, df: pd.DataFrame, preds: list[dict]) -> None:
        rows = [{"features": r.to_dict(), "prob": x["fraud_probability"], "anomaly": x["anomaly_score"],
                 "score": x["decision_score"], "risk": x["risk_category"], "flagged": x["decision"] == "REVIEW"}
                for (_, r), x in zip(df.iterrows(), preds, strict=True)]
        st.store.add_events(rows, "live", None, p.version)

    @app.post("/predict", response_model=Prediction)
    def predict(tx: Transaction, request: Request):  # type: ignore[valid-type]
        st = S(request)
        p = st.require_model()
        df = pd.DataFrame([tx.model_dump()])
        pred = p.predict_frame(df)
        _log_events(st, p, df, pred)
        return pred[0]

    @app.post("/predict/batch")
    def predict_batch(body: BatchRequest, request: Request):
        st = S(request)
        p = st.require_model()
        df = pd.DataFrame([t.model_dump() for t in body.transactions])
        preds = p.predict_frame(df)
        _log_events(st, p, df, preds)
        return {"model_version": p.version, "count": len(preds), "predictions": preds}

    # ------------------------------------------------------------------ monitoring
    @app.post("/monitor/events")
    def monitor_events(body: EventsRequest, request: Request):
        st = S(request)
        p = st.require_model()
        df = pd.DataFrame([e.model_dump(exclude={"label"}) for e in body.events])
        preds = p.predict_frame(df)
        rows = [{"features": r.to_dict(), "prob": x["fraud_probability"], "anomaly": x["anomaly_score"],
                 "score": x["decision_score"], "risk": x["risk_category"], "flagged": x["decision"] == "REVIEW"}
                for (_, r), x in zip(df.iterrows(), preds, strict=True)]
        n = st.store.add_events(rows, body.source, None, p.version, labels=[e.label for e in body.events])
        return {"accepted": n, "source": body.source, "model_version": p.version}

    @app.get("/monitor/status")
    def monitor_status(request: Request):
        st = S(request)
        return {"latest_report": st.monitor.latest(), "event_stats": st.store.event_stats(),
                "recent_reports": [{k: r.get(k) for k in ("id", "created_at", "status", "drift_detected", "batch_tag",
                                                           "contains_simulated_events")}
                                   | {"share_drifted": (r.get("feature_drift") or {}).get("share_drifted")}
                                   for r in st.store.list_reports(10)]}

    def _maybe_trigger(st: AppState, summary: dict, auto: bool | None) -> dict | None:
        auto = st.settings.auto_retrain_on_drift if auto is None else auto
        if not (auto and summary.get("drift_detected")):
            return None
        try:
            run = st.orchestrator.trigger("drift", reason=summary.get("drift_reason"), drift_report_id=summary["id"])
            return {"triggered": True, "run_id": run["id"]}
        except (RetrainingConflict, RetrainingRejected) as e:
            return {"triggered": False, "reason": str(e)}

    @app.post("/monitor/evaluate")
    def monitor_evaluate(body: EvaluateRequest, request: Request):
        st = S(request)
        summary = st.monitor.evaluate(st.require_model(), body.window, body.source)
        return {"report": summary, "retraining": _maybe_trigger(st, summary, body.auto_trigger)}

    @app.post("/monitor/simulate")
    def monitor_simulate(body: SimulateRequest, request: Request):
        st = S(request)
        try:
            res = run_simulation(st.monitor, st.store, st.require_model(), body.n, body.features, body.magnitude,
                                 body.amount_factor, body.seed)
        except ValueError as e:
            raise HTTPException(422, detail=str(e)) from e
        res["retraining"] = _maybe_trigger(st, res["after"], body.auto_trigger)
        return res

    @app.get("/monitor/reports/{report_id}/html")
    def monitor_report_html(report_id: str, request: Request, kind: str = "features"):
        rep = S(request).store.get_report(report_id)
        if not rep or rep.get("status") != "ok":
            raise HTTPException(404, detail="report not found or has no HTML")
        path = rep["paths"]["feature_html" if kind == "features" else "prediction_html"]
        return FileResponse(path, media_type="text/html")

    # ------------------------------------------------------------------ retraining
    @app.get("/retraining/status")
    def retraining_status(request: Request, limit: int = 20):
        st = S(request)
        return {"active_model": st.registry.active(), "active_run": bool(st.store.active_runs()),
                "runs": st.store.list_runs(min(max(limit, 1), 100))}

    @app.post("/retraining/trigger", status_code=202)
    def retraining_trigger(body: RetrainRequest, request: Request):
        st = S(request)
        try:
            return st.orchestrator.trigger("manual", body.reason or "manual request", body.overrides)
        except RetrainingConflict as e:
            raise HTTPException(409, detail=str(e)) from e
        except (RetrainingRejected, ValueError) as e:
            raise HTTPException(422, detail=str(e)) from e

    @app.post("/retraining/rollback")
    def retraining_rollback(body: RollbackRequest, request: Request):
        st = S(request)
        try:
            target = st.orchestrator.rollback(body.reason, body.to_version)
        except RetrainingConflict as e:
            raise HTTPException(409, detail=str(e)) from e
        except (RegistryError, ArtifactError) as e:
            raise HTTPException(422, detail=str(e)) from e
        return {"active_model": target}

    # ------------------------------------------------------------------ metrics
    @app.get("/metrics")
    def metrics(request: Request):
        st = S(request)
        lat = np.array(st.latencies) if st.latencies else None
        return {
            "requests_total": st.counts["requests_total"], "errors_total": st.counts["errors_total"],
            "by_endpoint": {k: v for k, v in st.counts.items() if " /" in k},
            "errors_by_status": {k: v for k, v in st.counts.items() if k.startswith("errors_") and k != "errors_total"},
            "predict_latency_ms": None if lat is None else {
                "n": int(len(lat)), "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                "p99": float(np.percentile(lat, 99)), "note": "server-side, in-process window of last 5000 requests"},
            "predictions": st.store.event_stats(), "model_version": st.predictor.version if st.predictor else None,
        }

    return app


app = create_app()  # settings come from SENTINEL_* env vars; filesystem is only touched at startup
