# Architecture

**Components** (all under `app/`): `data` (validation, split, synthetic test generator) · `features` (preprocessing) · `models` (training, scoring, artifacts, predictor, metrics)
· `registry` (active pointer + approved stack) · `monitoring` (event store, Evidently drift, simulation) · `retraining` (state machine, gate) · `api` (FastAPI) · `dashboard` (static HTML/CSS/JS served by the API at `/`, HTTP-only).

**Artifact** `var/models/<version>/`: `model.joblib` (preprocessor, baseline, XGBoost, IsolationForest, score normaliser), `manifest.json` (version, threshold, feature schema,
dataset info, training config, metrics, library versions, SHA-256 of every file), `metrics.json`, `config.json`, `reference.csv.gz` (validation sample + scores), `pool.csv.gz` (test feature sample for simulation), `examples.json`.

**Request flow:** `/predict` → Pydantic validation (exact feature set, ranges) → `Predictor.predict_frame` (same preprocessing object as training) → response + event row (source `live`).

**Monitoring flow:** events table → last N rows (filtered by source/batch/model version) → Evidently vs the *active model's* reference → report row + HTML → `drift_condition()` (pure function).

**Retraining flow:** trigger → run row (`pending`) → worker thread: checks → train candidate (artifact + MLflow run) → score candidate and active on the same validation split →
`evaluate_gate` → promote (registry + hot swap) or reject. The API process owns the worker; run exactly one uvicorn worker.

**Storage:** SQLite (`events`, `monitor_reports`, `retraining_runs`), `registry.json` (atomic replace), artifact folders, MLflow SQLite DB.
