# Troubleshooting

* **`Dataset not found`** → place `creditcard.csv` at `data/raw/` or set `SENTINEL_DATA_PATH`.
* **API `/health` says `model_loaded: false`** → run `python scripts/train.py`, then restart the API (or trigger a promotion).
* **`Split 'x' has only N fraud cases`** → dev sample too small; raise `SENTINEL_SAMPLE_FRAC`.
* **`Checksum mismatch`** → the artifact was modified/corrupted; retrain or roll back to another version.
* **MLflow import error mentioning `FallbackAsyncAdaptedQueuePool`** → SQLAlchemy too new; install `sqlalchemy==2.0.36`. Training continues and logs `mlflow_tracking_failed` if tracking fails.
* **`evidently` import errors / numpy 2.x** → Evidently 0.4.40 needs `numpy<2.1`; reinstall from `requirements.txt` in a clean venv.
* **Docker: permission denied writing `/app/var`** → the container runs as uid 1000; `mkdir -p var && chmod 777 var` or `sudo chown 1000 var`.
* **Dashboard shows "API unreachable"** → the dashboard is served by the API at `http://localhost:8000/`; check that `make serve` or `docker compose up` is running.
* **409 on retrain** → a run is active; wait or check `/retraining/status`. After a crash the orphaned run is marked failed at the next startup.
* **Insufficient data on evaluate** → send/simulate at least `SENTINEL_MONITOR_MIN_SAMPLES` events.
* **Slow first request** → XGBoost/sklearn warm-up; the benchmark warms up before timing.
