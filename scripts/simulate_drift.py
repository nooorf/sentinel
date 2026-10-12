"""Run the controlled drift simulation against the active model and save before/after reports (offline, no API)."""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401

from app.common.config import Settings
from app.models.predictor import Predictor
from app.monitoring.service import MonitoringService
from app.monitoring.simulate import run_simulation
from app.monitoring.store import Store
from app.registry.registry import ModelRegistry

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--magnitude", type=float, default=2.0)
ap.add_argument("--amount-factor", type=float, default=3.0)
a = ap.parse_args()
s = Settings.from_env()
reg = ModelRegistry(s.models_dir)
if not reg.active():
    raise SystemExit("No active model. Run scripts/train.py first.")
pred = Predictor.load(reg.artifact_path(reg.active()))
store = Store(s.db_path)
res = run_simulation(MonitoringService(s, store), store, pred, n=a.n, magnitude=a.magnitude, amount_factor=a.amount_factor)
for k in ("before", "after"):
    r = res[k]
    fd = r.get("feature_drift", {})
    print(f"{k:>6}: status={r['status']} drifted={fd.get('n_drifted')}/{fd.get('n_features')} "
          f"share={fd.get('share_drifted')} dataset_drift={r.get('drift_detected')} "
          f"prediction_drift={r.get('prediction_drift', {}).get('drift_detected')}")
out = Path("reports/drift_simulation.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(res, indent=1, default=str))
print(f"saved {out}; HTML reports in {s.reports_dir / 'drift'}")
