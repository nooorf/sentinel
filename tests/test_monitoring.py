from app.monitoring.drift import DriftConfig, drift_condition, performance_summary


def ok_summary(share, pred=False):
    return {"status": "ok", "feature_drift": {"share_drifted": share, "n_drifted": int(share * 29), "n_features": 29},
            "prediction_drift": {"drift_detected": pred}}


def test_drift_condition_thresholds():
    cfg = DriftConfig(share_threshold=0.3)
    assert drift_condition(ok_summary(0.4), cfg)[0] is True
    assert drift_condition(ok_summary(0.1), cfg)[0] is False
    assert drift_condition(ok_summary(0.1, pred=True), cfg)[0] is False
    assert drift_condition(ok_summary(0.1, pred=True), DriftConfig(prediction_drift_triggers=True))[0] is True
    assert drift_condition({"status": "insufficient_data"}, cfg)[0] is False


def test_performance_unavailable_without_labels(client, tx):
    client.post("/monitor/events", json={"events": [tx] * 5})
    import pandas as pd
    assert performance_summary(pd.DataFrame({"label": [None, None], "flagged": [0, 1]}), 30)["status"] == "unavailable"


def test_insufficient_data_is_reported_not_faked(client, tx):
    client.post("/predict", json=tx)
    r = client.post("/monitor/evaluate", json={}).json()["report"]
    assert r["status"] == "insufficient_data" and r["drift_detected"] is None


def test_real_evidently_report_and_simulation_changes_result(client):
    sim = client.post("/monitor/simulate", json={"n": 300, "magnitude": 3.0}).json()
    before, after = sim["before"], sim["after"]
    assert before["status"] == "ok" and after["status"] == "ok"
    assert after["feature_drift"]["share_drifted"] > before["feature_drift"]["share_drifted"]
    assert after["drift_detected"] is True and before["drift_detected"] is False
    assert after["contains_simulated_events"] and "SIMULATED" in sim["label"]
    html = client.get(f"/monitor/reports/{after['id']}/html")
    assert html.status_code == 200 and "evidently" in html.text.lower()
    st = client.get("/monitor/status").json()
    assert st["latest_report"]["id"] == after["id"]
    assert st["latest_report"]["performance"]["status"] == "unavailable"  # simulated events carry no labels
