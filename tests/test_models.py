import numpy as np
import pytest

from app.data.schema import FEATURES, chronological_split
from app.data.synthetic import make_synthetic_creditcard
from app.models.artifacts import ArtifactError, load_artifact
from app.models.metrics import compute_metrics, select_threshold
from app.models.predictor import InputValidationError, Predictor
from app.models.train import TrainConfig, train_pipeline


def test_threshold_meets_target_recall_and_maximises_precision():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    s = np.clip(rng.normal(0.3, 0.15, 5000) + 0.4 * y, 0, 1)
    thr, info = select_threshold(y, s, 0.8)
    m = compute_metrics(y, s, thr)
    assert m["recall"] >= 0.8 and not info["fallback"]
    for t in np.linspace(thr, 1, 40)[1:]:  # any higher threshold either loses recall or does not beat precision
        mm = compute_metrics(y, s, t)
        assert mm["recall"] < 0.8 or mm["precision"] <= m["precision"] + 1e-9


def test_metrics_confusion_consistency():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1) and m["false_positive_count"] == 1


def test_training_report_and_no_test_leakage():
    sp = chronological_split(make_synthetic_creditcard(5000, 0.05, seed=5))
    b = train_pipeline(sp, TrainConfig(n_estimators=30, bootstrap=10))
    r = b["report"]
    assert set(r["models"]) == {"baseline", "xgboost", "iforest", "ensemble"}
    # threshold equals the one stored from validation selection (test never tunes it)
    assert b["threshold"] == r["models"][b["decision_model"]]["threshold"]
    assert r["models"]["xgboost"]["validation"]["recall"] >= 0.8 - 1e-9
    assert r["models"]["xgboost"]["test"]["pr_auc"] > 0.5  # learnable synthetic signal
    assert "conclusion" in r["ensemble_analysis"]


def test_ensemble_dropped_when_gain_too_small():
    sp = chronological_split(make_synthetic_creditcard(5000, 0.05, seed=6))
    b = train_pipeline(sp, TrainConfig(n_estimators=30, bootstrap=5, ensemble_min_gain=10.0))
    assert b["decision_model"] == "xgboost" and b["ensemble_weight"] == 1.0


def test_artifact_manifest_and_prediction(env):
    s, v = env
    p = Predictor.load(s.models_dir / v)
    m = p.manifest
    for k in ("version", "threshold", "features", "dataset", "training_config", "created_at", "model_sha256", "headline_metrics"):
        assert k in m
    out = p.predict_frame(__import__("pandas").DataFrame([{f: 0.0 for f in FEATURES}]))[0]
    assert out["model_version"] == v and 0 <= out["fraud_probability"] <= 1 and 0 <= out["anomaly_score"] <= 1
    assert out["risk_category"] in {"LOW", "MEDIUM", "HIGH"}


def test_tampered_artifact_refused(env):
    s, v = env
    f = s.models_dir / v / "model.joblib"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(ArtifactError, match="Checksum"):
        load_artifact(s.models_dir / v)


def test_predictor_rejects_missing_and_extra_features(env):
    import pandas as pd
    s, v = env
    p = Predictor.load(s.models_dir / v)
    with pytest.raises(InputValidationError):
        p.predict_frame(pd.DataFrame([{f: 0.0 for f in FEATURES[:-1]}]))
    with pytest.raises(InputValidationError):
        p.predict_frame(pd.DataFrame([{**{f: 0.0 for f in FEATURES}, "Class": 1}]))


def test_value_metrics():
    from app.models.metrics import value_metrics
    y = np.array([1, 1, 0, 0, 0])
    m = value_metrics(y, np.array([0.9, 0.1, 0.8, 0.2, 0.1]), 0.5, np.array([100.0, 50.0, 30.0, 10.0, 5.0]), review_cost=2)
    assert m["flagged"] == 2 and m["fraud_value_caught"] == 100 and m["fraud_value_missed"] == 50
    assert m["legit_value_flagged"] == 30 and m["total_cost"] == 50 + 2 * 2 and abs(m["fraud_value_caught_share"] - 2 / 3) < 1e-9
