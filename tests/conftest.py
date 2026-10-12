"""Fixtures use SYNTHETIC data (tests only). A model is trained once per session and copied per test."""
import shutil

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.common.config import Settings
from app.data.schema import FEATURES
from app.data.synthetic import make_synthetic_creditcard
from app.models.pipeline import train_and_save
from app.registry.registry import ModelRegistry


def base_settings(csv, data_dir, **kw) -> Settings:
    return Settings(data_path=csv, data_dir=data_dir, monitor_min_samples=100, retrain_min_fraud=5,
                    gate_min_pr_auc_delta=-0.05, **kw)


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "synthetic.csv"
    make_synthetic_creditcard(6000, seed=0).to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory, synthetic_csv):
    d = tmp_path_factory.mktemp("trained")
    s = base_settings(synthetic_csv, d)
    v, *_ = train_and_save(s, {"n_estimators": 40, "bootstrap": 20})
    reg = ModelRegistry(s.models_dir)
    reg.register_candidate(v, {}, "test")
    reg.promote(v, "initial test model")
    return d, v


@pytest.fixture
def env(tmp_path, synthetic_csv, trained_dir):
    """Fresh, isolated copy of the trained state."""
    src, v = trained_dir
    dst = tmp_path / "var"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("sentinel.db*", "mlflow.db"))
    return base_settings(synthetic_csv, dst), v


@pytest.fixture
def client(env):
    s, _ = env
    with TestClient(create_app(s)) as c:
        yield c


@pytest.fixture
def tx():
    return {f: 0.0 for f in FEATURES}
