.PHONY: install check-data train-dev train serve test lint simulate benchmark up down smoke
PY ?= python
install:
	$(PY) -m pip install -r requirements-dev.txt
check-data:
	$(PY) scripts/check_data.py
train-dev:        ## quick model on a subsample (all fraud rows + 10% of legit rows)
	SENTINEL_SAMPLE_FRAC=0.1 $(PY) scripts/train.py
train:            ## full dataset
	$(PY) scripts/train.py
serve:
	$(PY) -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
simulate:
	$(PY) scripts/simulate_drift.py
benchmark:
	$(PY) scripts/benchmark.py --spawn
test:
	$(PY) -m pytest
lint:
	$(PY) -m ruff check .
smoke:
	$(PY) scripts/smoke_test.py
up:
	docker compose up --build -d
down:
	docker compose down
