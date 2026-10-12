<div align="center">

<img src="docs/assets/hero.svg" alt="Sentinel: real-time fraud detection that watches itself. Transactions are scored live against a review threshold and fraud is sent to review." width="100%">

<br>

[![CI](https://img.shields.io/github/actions/workflow/status/nooorf/sentinel/ci.yml?branch=main&style=for-the-badge&label=CI&labelColor=33302E&logo=githubactions&logoColor=white)](https://github.com/nooorf/sentinel/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-0F5499?style=for-the-badge&labelColor=33302E&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-0D7680?style=for-the-badge&labelColor=33302E&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-990F3D?style=for-the-badge&labelColor=33302E)<br>
![Evidently](https://img.shields.io/badge/Evidently-drift-E8812A?style=for-the-badge&labelColor=33302E)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0B4580?style=for-the-badge&labelColor=33302E&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-one%20container-0F5499?style=for-the-badge&labelColor=33302E&logo=docker&logoColor=white)

**[Demo](#demo) · [Results](#results) · [Architecture](#architecture) · [Quick start](#quick-start) · [API](#api)**

</div>

An end-to-end ML system for card fraud: an XGBoost and Isolation Forest scoring API, Evidently drift monitoring, a model registry with a promotion gate and rollback, and a fraud-operations console, all shipped as one Docker container.

## Demo

<div align="center">
<img src="docs/assets/demo.gif" alt="Sentinel console: scoring a fraudulent and a legitimate held-out transaction, simulating a traffic shift that raises a drift alert, and the model registry" width="100%">
<br>
<sub>Scoring real held-out transactions, raising a drift alert on a simulated traffic shift, and the model registry. Everything is served live by the API.</sub>
</div>

## Results

<img src="docs/assets/results.svg" alt="On 56,746 held-out transactions Sentinel cuts false alarms from 91 to 49 (−46%), manual reviews from 153 to 106 (−31%) and legitimate spend held from €66,282 to €35,420 (−47%). Precision 0.538 vs 0.405, PR-AUC 0.776 vs 0.744, fraud cost −62% vs no screening, p99 latency 9.7 ms, 260 requests/s." width="100%">

Evaluated on a **held-out, time-ordered test window of 56,746 real card transactions** that the model never saw during training or tuning, against a logistic-regression baseline. Both models' thresholds were tuned on validation data to an 80% recall target. Fraud cost is missed fraud value plus €5 per manual review, compared with letting every transaction through.

<details>
<summary><b>Latency breakdown</b>: 2 ms model inference, under 10 ms p99 end to end over HTTP</summary>
<br>

| Apple M-series laptop, 1 worker | p50 | p95 | p99 |
|---|---:|---:|---:|
| Model inference, single transaction | 1.9 ms | 2.0 ms | 2.1 ms |
| `POST /predict` end to end over HTTP | 3.8 ms | 4.2 ms | **9.7 ms** |
| `POST /predict/batch`, 100 transactions | 10.9 ms | 11.0 ms | 11.1 ms |
| Throughput at concurrency 8 | **260 requests/s** | | |

</details>

Full reports: [model evaluation](reports/model_evaluation_report.md) · [benchmark](docs/benchmarks.md)

## What it does

<table>
<tr>
<td width="50%" valign="top">

### ⚡ Scores in real time
`POST /predict` validates the payload, applies the training-time preprocessing, and returns a fraud probability, an anomaly rank, a risk level and an APPROVE / REVIEW decision in under 10 ms.

</td>
<td width="50%" valign="top">

### 📈 Watches its own inputs
Every scored transaction is logged. Evidently compares recent traffic with the serving model's reference data feature by feature, and also checks for prediction drift.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🔁 Retrains itself, safely
Drift alerts or a manual request start a background retraining run. The candidate is promoted only if it passes a validation gate against the serving model, and is otherwise rejected and kept for audit.

</td>
<td width="50%" valign="top">

### 🗂️ Gives the fraud team a console
A purpose-built dashboard served by the same API shows impact, live scoring, drift checks and the model registry with one-click rollback.

</td>
</tr>
</table>

## Architecture

<img src="docs/assets/architecture.svg" alt="Architecture: offline training (validate and time-ordered split, train LR, XGBoost and Isolation Forest, versioned SHA-256 artifact, MLflow) feeds the model registry. In the service, the ops console calls POST /predict, predictions go to the event store, Evidently checks drift, an alert starts retraining, and a promotion gate either promotes the candidate (hot swap, no restart) or rejects it." width="100%">

## Engineering highlights

<details>
<summary><b>Machine learning done rigorously</b>: leakage-free time split, imbalance-aware XGBoost, cost-aware threshold, bootstrap confidence intervals</summary>
<br>

- **No leakage by construction.** The split is chronological (60/20/20 by transaction time). Preprocessing is fitted on the training data only. Thresholds, ensemble weights and model selection use validation data only, and the test split is scored exactly once.
- **Built for extreme imbalance (0.17% fraud).** XGBoost uses `scale_pos_weight` instead of resampling, early-stops on validation PR-AUC, and is evaluated on PR-AUC, precision and recall rather than accuracy.
- **Business-aware operating point.** The threshold maximises precision subject to a recall target. Results are reported both statistically and in money: fraud value caught, legitimate spend held, and total cost.
- **Uncertainty reported.** Every test metric comes with 95% bootstrap confidence intervals.
- **Automated model selection.** XGBoost is compared with XGBoost + Isolation Forest blends on validation PR-AUC. The simpler model serves unless a blend wins by a set margin.

</details>

<details>
<summary><b>MLOps</b>: tamper-evident artifacts, registry with promotion gate and rollback, drift-triggered retraining, MLflow tracking</summary>
<br>

- **Versioned, tamper-evident artifacts.** Each version bundles the model, preprocessor, threshold, drift reference, metrics and a SHA-256 manifest, which is verified before every load or promotion.
- **Registry with an audit trail.** It tracks candidate, approved, rejected and rolled-back versions. Promotion hot-swaps the serving model with no restart, and rollback re-verifies the previous artifact first.
- **Promotion gate.** The candidate and the serving model are scored on the same validation split. A candidate is promoted only if it clears recall and precision floors, has at least 30 validation frauds behind it, and does not lose PR-AUC.
- **Drift monitoring with Evidently.** Per-feature drift and prediction drift are computed against the serving model's own reference. HTML reports are stored, and alerts can trigger retraining automatically.
- **Retraining state machine.** Runs move from `pending` to `checking`, `training` and `evaluating`, then end as `promoted`, `rejected`, `failed` or `skipped`. A single worker prevents concurrent runs, and runs orphaned by a crash are marked failed at startup.
- **Experiment tracking.** Every training run logs its parameters and metrics to MLflow.

</details>

<details>
<summary><b>Software engineering</b>: strict Pydantic API, 42 tests, CI, one-container deploy, reproducible benchmark, Azure ML scripts</summary>
<br>

- **FastAPI with strict Pydantic v2 schemas.** Unknown, missing, non-finite or out-of-range fields return HTTP 422, and nothing is silently defaulted.
- **42 automated tests** cover validation, splitting, leakage invariants, artifact tamper detection, API contracts, drift triggers, promotion and rejection paths, rollback, and drift-to-retraining end to end.
- **CI on GitHub Actions** runs lint (ruff), the tests, a smoke test and a Docker build.
- **One container** serves the API, the console and the retraining worker. Model state persists on a volume, and a health check is built in.
- **A reproducible latency benchmark** reports p50, p95 and p99 for model-only, HTTP, batch and concurrent load.
- **Cloud deployment scripts** for Azure ML managed online endpoints, with blue/green slots, a smoke test and traffic-switch rollback.

</details>

## The operations console

| Score a transaction | Drift monitoring | Model registry |
|---|---|---|
| ![Score](docs/screenshots/score.png) | ![Monitoring](docs/screenshots/monitoring.png) | ![Models](docs/screenshots/models.png) |
| Real held-out transactions scored live, with the decision placed on the fraud-probability scale against the review threshold. | A simulated traffic shift, Evidently checks on every feature, alert status, and the full reports. | Every version and retraining run, a one-click retrain through the promotion gate, and rollback. |

The console is hand-built HTML, CSS and JavaScript with charts drawn in SVG. It has no frontend framework and no CDN dependency, and every figure on it is served live by the API.

## Quick start

**Prerequisites:** Python 3.11 and Docker. Download `creditcard.csv` from the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `data/raw/`.

```bash
# 1. Install
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # macOS: brew install libomp (needed by XGBoost)

# 2. Train, evaluate, benchmark
python scripts/train.py                      # trains all models, registers and activates the best one
python scripts/evaluate.py                   # writes reports/model_evaluation_report.md
python scripts/benchmark.py --spawn          # writes reports/benchmark.md

# 3. Run: console at http://localhost:8000, API docs at http://localhost:8000/docs
make serve
```

**With Docker:**

```bash
python scripts/train.py                      # or train inside the container (see below)
docker compose up -d --build                 # console + API on http://localhost:8000
# docker compose run --rm api python scripts/train.py
```

**Tests:** `make test` · `make lint` · `make smoke`

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score one transaction: fraud probability, anomaly rank, risk level, decision |
| `POST /predict/batch` | Score up to 500 transactions |
| `GET /model/info` · `/model/evaluation` · `/model/versions` | Serving model metadata, full evaluation, registry |
| `POST /monitor/events` · `/monitor/evaluate` · `/monitor/simulate` | Ingest events (with optional delayed labels), run a drift check, simulate a shift |
| `GET /monitor/status` · `/monitor/reports/{id}/html` | Latest drift status and Evidently HTML reports |
| `POST /retraining/trigger` · `/retraining/rollback` · `GET /retraining/status` | Start a gated retraining run, roll back, view run history |
| `GET /health` · `/metrics` | Health check; request counts and live latency percentiles |

```bash
# examples/transaction.json is a real fraudulent transaction from the held-out test window
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @examples/transaction.json
# {"fraud_probability": 0.9969, "anomaly_score": 0.98, "risk_category": "HIGH", "decision": "REVIEW", ...}
```

<details>
<summary><b>Project structure</b></summary>
<br>

```
app/
  api/          FastAPI service, request/response schemas
  dashboard/    operations console (static HTML/CSS/JS, served at /)
  data/         schema validation, time-ordered splitting
  features/     preprocessing pipeline (fitted on training data only)
  models/       training, scoring, metrics, versioned artifacts
  monitoring/   event store, Evidently drift checks, simulation
  registry/     model registry: promotion, rejection, rollback
  retraining/   orchestrator state machine and promotion gate
scripts/        train · evaluate · benchmark · simulate_drift · smoke_test
examples/       sample transaction for the API
tests/          42 tests on synthetic fixtures (no dataset needed)
deployment/     Azure ML managed online endpoint scripts
docs/           architecture, engineering decisions, evaluation, benchmarks
```

</details>

More detail: [architecture](docs/architecture.md) · [engineering decisions](docs/engineering_decisions.md) · [model evaluation](docs/model_evaluation.md) · [benchmarks](docs/benchmarks.md) · [Azure ML deployment](docs/azure_ml.md)

## Tech stack

Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · Evidently · FastAPI · Pydantic v2 · Uvicorn · MLflow · SQLite · Docker Compose · GitHub Actions · pytest · ruff · Azure ML SDK v2

## Data

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the Machine Learning Group of ULB and Worldline: 284,807 European card transactions over two days, 0.17% fraud. Features `V1`–`V28` are PCA components; `Amount` is in EUR. Database Contents License (DbCL) v1.0. The data is not redistributed in this repository.
