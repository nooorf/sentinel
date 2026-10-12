"""Inspect registry state and the active model's manifest/metrics."""
import json

import _common  # noqa: F401

from app.common.config import Settings
from app.models.artifacts import read_manifest
from app.registry.registry import ModelRegistry

s = Settings.from_env()
reg = ModelRegistry(s.models_dir)
snap = reg.snapshot()
print("ACTIVE:", snap["active"])
for v, r in snap["versions"].items():
    print(f"  {v}  status={r['status']}  registered={r['registered_at']}")
if snap["active"]:
    m = read_manifest(reg.artifact_path(snap["active"]))
    print(json.dumps({k: m[k] for k in ("version", "created_at", "decision_model", "threshold", "dataset", "headline_metrics")}, indent=2))
