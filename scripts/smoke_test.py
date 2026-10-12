"""Lightweight CI smoke test on SYNTHETIC data: train -> serve (in-process) -> health + predict + monitoring."""
import os
import tempfile

import _common  # noqa: F401

tmp = tempfile.mkdtemp()
os.environ["SENTINEL_DATA_DIR"] = tmp
os.environ["SENTINEL_DATA_PATH"] = os.path.join(tmp, "synthetic.csv")
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.common.config import Settings
from app.data.schema import FEATURES
from app.data.synthetic import make_synthetic_creditcard
from app.models.pipeline import train_and_save
from app.registry.registry import ModelRegistry

make_synthetic_creditcard(6000).to_csv(os.environ["SENTINEL_DATA_PATH"], index=False)
s = Settings.from_env().with_overrides(monitor_min_samples=100)
v, *_ = train_and_save(s, {"n_estimators": 40})
reg = ModelRegistry(s.models_dir); reg.register_candidate(v, {}, "smoke"); reg.promote(v, "smoke")
with TestClient(create_app(s)) as c:
    assert c.get("/health").json()["model_loaded"]
    r = c.post("/predict", json={f: 0.0 for f in FEATURES}); assert r.status_code == 200, r.text
    sim = c.post("/monitor/simulate", json={"n": 200}); assert sim.status_code == 200, sim.text
    print("SMOKE OK", v, r.json()["decision"], sim.json()["after"]["status"])
