"""Train the initial model, save a versioned artifact and register it as the active model."""
import argparse
import json

import _common  # noqa: F401

from app.common.config import Settings
from app.models.pipeline import train_and_save
from app.registry.registry import ModelRegistry

ap = argparse.ArgumentParser()
ap.add_argument("--n-estimators", type=int)
ap.add_argument("--no-activate", action="store_true", help="register as candidate only")
a = ap.parse_args()
s = Settings.from_env()
ov = {"n_estimators": a.n_estimators} if a.n_estimators else None
version, path, bundle, splits = train_and_save(s, ov, run_name="initial-train")
reg = ModelRegistry(s.models_dir)
m = bundle["report"]["models"][bundle["decision_model"]]
reg.register_candidate(version, {"validation": m["validation"], "test": m["test"]}, "trained by scripts/train.py")
if not a.no_activate:
    reg.promote(version, "initial model (scripts/train.py)")
print(f"version={version}\npath={path}\ndecision_model={bundle['decision_model']}\nactive={reg.active()}")
print(json.dumps({"validation": m["validation"], "test": m["test"]}, indent=2))
