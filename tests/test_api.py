from app.data.schema import FEATURES


def test_health_and_info(client, env):
    _, v = env
    h = client.get("/health").json()
    assert h["model_loaded"] and h["model_version"] == v
    info = client.get("/model/info").json()
    assert info["version"] == v and info["features"] == FEATURES


def test_predict_schema_and_version(client, env, tx):
    _, v = env
    r = client.post("/predict", json=tx)
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {"fraud_probability", "anomaly_score", "decision_score", "risk_category", "decision",
                      "decision_model", "model_version"}
    assert b["model_version"] == v and b["decision"] in {"APPROVE", "REVIEW"}


def test_predict_missing_field_422(client, tx):
    tx.pop("V3")
    r = client.post("/predict", json=tx)
    assert r.status_code == 422 and "V3" in r.text


def test_predict_unknown_field_and_bad_range_422(client, tx):
    assert client.post("/predict", json={**tx, "Class": 1}).status_code == 422
    assert client.post("/predict", json={**tx, "Amount": -1}).status_code == 422
    assert client.post("/predict", json={**tx, "V1": "abc"}).status_code == 422
    assert client.post("/predict", json={**tx, "V1": 1e9}).status_code == 422


def test_batch_validates_every_row_and_bounds(client, tx):
    ok = client.post("/predict/batch", json={"transactions": [tx, tx]})
    assert ok.status_code == 200 and ok.json()["count"] == 2
    bad = tx.copy(); bad.pop("V1")
    assert client.post("/predict/batch", json={"transactions": [tx, bad]}).status_code == 422
    assert client.post("/predict/batch", json={"transactions": [tx] * 501}).status_code == 422
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_metrics_endpoint_counts_requests_and_errors(client, tx):
    client.post("/predict", json=tx)
    client.post("/predict", json={})
    m = client.get("/metrics").json()
    assert m["errors_total"] >= 1 and m["predict_latency_ms"]["n"] >= 2
    assert m["predictions"]["live_predictions"] == 1


def test_monitor_events_bounded_and_labels_optional(client, tx):
    r = client.post("/monitor/events", json={"events": [tx, {**tx, "label": 1}]})
    assert r.status_code == 200 and r.json()["accepted"] == 2
    assert client.post("/monitor/events", json={"events": [{**tx, "label": 5}]}).status_code == 422
    assert client.post("/monitor/events", json={"events": [tx] * 1001}).status_code == 422


def test_no_model_returns_503(tmp_path, synthetic_csv, tx):
    from fastapi.testclient import TestClient

    from app.api.main import create_app
    from tests.conftest import base_settings
    with TestClient(create_app(base_settings(synthetic_csv, tmp_path / "empty"))) as c:
        assert c.get("/health").json() == {**c.get("/health").json(), "model_loaded": False}
        assert c.post("/predict", json=tx).status_code == 503


def test_dashboard_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "Sentinel" in r.text
    for path in ("/static/app.js", "/static/app.css", "/static/fonts/libre-franklin-latin.woff2"):
        assert client.get(path).status_code == 200, path


def test_evaluation_includes_money_view(client):
    ev = client.get("/model/evaluation").json()
    tv = ev["models"][ev["decision_model"]]["test_value"]
    assert {"flagged", "fraud_value_caught", "legit_value_flagged", "total_cost"} <= set(tv)
