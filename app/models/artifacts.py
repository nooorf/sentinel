"""Versioned model artifacts.

Layout:  <models_dir>/<version>/{model.joblib, manifest.json, metrics.json, config.json,
                                   reference.csv.gz, pool.csv.gz, examples.json}

SECURITY: joblib/pickle can execute arbitrary code on load. Only load artifacts produced by this project on a
machine you control. `load_artifact` verifies the SHA-256 recorded in manifest.json to detect corruption or
accidental modification; it is NOT a defence against an attacker who can also rewrite the manifest.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from app.common.jsonutil import dumps, utcnow
from app.data.schema import FEATURES, LABEL
from app.models.scoring import score_parts

SCHEMA_VERSION = 1
MODEL_FILE = "model.joblib"


class ArtifactError(RuntimeError):
    pass


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifact(bundle: dict, splits, dataset_info: dict, models_dir: Path, seed: int = 42) -> tuple[str, Path]:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    tmp = models_dir / f".tmp-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        slim = {k: v for k, v in bundle.items() if k not in ("report", "config")}
        joblib.dump(slim, tmp / MODEL_FILE, compress=3)
        digest = _sha256(tmp / MODEL_FILE)
        version = f"v{utcnow().replace('-', '').replace(':', '').replace('T', '')[:14]}-{digest[:8]}"

        # reference = validation sample scored by this model; pool = test sample (for monitoring demos)
        ref = splits.val.sample(n=min(5000, len(splits.val)), random_state=seed)[FEATURES].reset_index(drop=True)
        ref["score"] = score_parts(bundle, ref)["decision"]
        ref.to_csv(tmp / "reference.csv.gz", index=False)
        pool = splits.test.sample(n=min(3000, len(splits.test)), random_state=seed)[FEATURES]
        pool.reset_index(drop=True).to_csv(tmp / "pool.csv.gz", index=False)

        # demo examples: real held-out rows the serving model handles correctly (caught fraud, approved legitimate)
        te = splits.test
        hit = score_parts(bundle, te)["decision"] >= bundle["threshold"]
        fr = te[(te[LABEL] == 1) & hit]
        lg = te[(te[LABEL] == 0) & ~hit]
        fr = fr if len(fr) else te[te[LABEL] == 1]
        lg = lg if len(lg) else te[te[LABEL] == 0]
        ex = pd.concat([fr.sample(n=min(3, len(fr)), random_state=seed), lg.sample(n=min(3, len(lg)), random_state=seed)])
        examples = [{"label": int(r[LABEL]), "note": "Real held-out test-set row (demonstration example)",
                     "features": {f: float(r[f]) for f in FEATURES}} for _, r in ex.iterrows()]
        (tmp / "examples.json").write_text(dumps(examples, indent=1))

        report = bundle["report"]
        (tmp / "metrics.json").write_text(dumps(report, indent=1))
        (tmp / "config.json").write_text(dumps(bundle["config"], indent=1))
        files = {p.name: _sha256(p) for p in sorted(tmp.iterdir()) if p.is_file()}
        manifest = {
            "schema_version": SCHEMA_VERSION, "version": version, "created_at": utcnow(),
            "decision_model": bundle["decision_model"], "threshold": bundle["threshold"],
            "ensemble_weight": bundle["ensemble_weight"], "features": FEATURES,
            "dataset": dataset_info, "splits": splits.info, "training_config": bundle["config"],
            "headline_metrics": {"validation": report["models"][bundle["decision_model"]]["validation"],
                                 "test": report["models"][bundle["decision_model"]]["test"]},
            "libraries": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                          "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
            "model_file": MODEL_FILE, "model_sha256": digest, "files_sha256": files,
            "contents": {MODEL_FILE: "preprocessor + baseline LR + XGBoost + IsolationForest + score normaliser",
                         "reference.csv.gz": "validation sample + model scores (drift reference)",
                         "pool.csv.gz": "test-split feature sample (drift simulation pool)",
                         "examples.json": "real test rows the model handles correctly, for the dashboard demo", "metrics.json": "full evaluation report"},
        }
        (tmp / "manifest.json").write_text(dumps(manifest, indent=1))
        final = models_dir / version
        if final.exists():
            raise ArtifactError(f"Version already exists: {version}")
        tmp.rename(final)
        return version, final
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_manifest(path: Path) -> dict:
    mf = Path(path) / "manifest.json"
    if not mf.exists():
        raise ArtifactError(f"manifest.json missing in {path}")
    return json.loads(mf.read_text())


def load_artifact(path: Path) -> tuple[dict, dict]:
    """Return (bundle, manifest) after verifying the model file checksum."""
    path = Path(path)
    manifest = read_manifest(path)
    model_path = path / manifest["model_file"]
    if not model_path.exists():
        raise ArtifactError(f"Model file missing: {model_path}")
    if _sha256(model_path) != manifest["model_sha256"]:
        raise ArtifactError(f"Checksum mismatch for {model_path}; refusing to load.")
    return joblib.load(model_path), manifest
