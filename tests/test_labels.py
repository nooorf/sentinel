"""Labelled monitoring events: chronological append into retraining data, per-model label counting. SYNTHETIC data."""
import numpy as np

from app.data.schema import FEATURES, LABEL, TIME_COL
from app.data.synthetic import make_synthetic_creditcard
from app.models.pipeline import prepare_splits
from app.monitoring.store import Store
from tests.conftest import base_settings


def test_labeled_events_are_appended_after_history(synthetic_csv, tmp_path):
    s = base_settings(synthetic_csv, tmp_path)
    ev = make_synthetic_creditcard(600, 0.2, seed=5)[[*FEATURES, LABEL]]
    splits, info = prepare_splits(s, ev)
    assert info["labeled_events_appended"] == 600 and info["labeled_event_frauds"] == int(ev[LABEL].sum())
    assert splits.test[TIME_COL].min() > splits.train[TIME_COL].max()  # still chronological
    # the newest rows are the events, in arrival order
    np.testing.assert_allclose(splits.test[FEATURES].tail(5).to_numpy(), ev[FEATURES].tail(5).to_numpy())


def test_count_labeled_is_per_model_version(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    row = {"features": {f: 0.0 for f in FEATURES}, "prob": 0.1, "anomaly": 0.1, "score": 0.1, "risk": "LOW", "flagged": False}
    store.add_events([row] * 3, "external", None, "v1", labels=[1, 1, 0])
    store.add_events([row] * 2, "external", None, "v2", labels=[1, None])
    store.add_events([row], "simulated", None, "v1", labels=[1])  # simulated events never count
    assert store.count_labeled("v1") == 2 and store.count_labeled("v2") == 1 and store.count_labeled("v1", 0) == 1
    assert len(store.labeled_events()) == 4
