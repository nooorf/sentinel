import pytest

from app.api.main import AppState
from app.retraining.orchestrator import Orchestrator, RetrainingConflict, RetrainingRejected, evaluate_gate


def make_orch(settings):
    st = AppState(settings)
    st.load_model()
    return st, st.orchestrator


def test_gate_policy_pure():
    from app.common.config import Settings
    s = Settings(gate_min_recall=0.7, gate_min_precision=0.3, gate_min_pr_auc_delta=0.0)
    good = {"recall": 0.8, "precision": 0.6, "pr_auc": 0.9, "n_fraud": 50}
    assert evaluate_gate(good, {"pr_auc": 0.89}, s)[0]
    assert not evaluate_gate({**good, "recall": 0.5}, None, s)[0]
    ok, why = evaluate_gate({**good, "pr_auc": 0.8}, {"pr_auc": 0.9}, s)
    assert not ok and "PR-AUC" in why[0]
    ok, why = evaluate_gate({**good, "n_fraud": 10, "pr_auc": 0.99}, {"pr_auc": 0.5}, s)  # big win on tiny evidence
    assert not ok and "insufficient evidence" in why[0]


def test_successful_promotion(env):
    s, v0 = env
    st, orch = make_orch(s)
    run = orch.trigger("manual", "test", {"n_estimators": 40, "bootstrap": 5, "seed": 7}, background=False)
    run = st.store.get_run(run["id"])
    assert run["status"] == "promoted", run
    v1 = run["candidate_version"]
    assert v1 != v0 and st.registry.active() == v1 and st.predictor.version == v1
    assert [t["status"] for t in run["transitions"]] == ["pending", "checking", "training", "evaluating", "promoted"]
    assert run["candidate_metrics"] and run["active_metrics"] and "promoted" in run["decision"]


def test_failed_candidate_rejected_and_active_model_kept(env):
    s, v0 = env
    strict = s.with_overrides(gate_min_recall=1.01)  # unreachable floor -> every candidate must be rejected
    st, orch = make_orch(strict)
    run = st.orchestrator.trigger("manual", "test", {"n_estimators": 40}, background=False)
    run = st.store.get_run(run["id"])
    assert run["status"] == "rejected" and "recall" in run["decision"]
    assert st.registry.active() == v0 and st.predictor.version == v0
    assert st.registry.snapshot()["versions"][run["candidate_version"]]["status"] == "rejected"
    assert (strict.models_dir / run["candidate_version"]).exists()  # kept for audit


def test_weak_candidate_rejected_by_pr_auc_gate(env):
    s, v0 = env
    st, orch = make_orch(s.with_overrides(gate_min_pr_auc_delta=0.0, gate_min_precision=0.0))
    run = orch.trigger("manual", "weak", {"n_estimators": 1, "max_depth": 1, "bootstrap": 5}, background=False)
    run = st.store.get_run(run["id"])
    assert run["status"] == "rejected" and st.registry.active() == v0, run["decision"]


def test_duplicate_concurrent_run_blocked(env):
    s, _ = env
    st, orch = make_orch(s)
    st.store.create_run({"id": "run-fake", "trigger": "manual", "status": "training", "requested_at": "now",
                         "transitions": []})
    with pytest.raises(RetrainingConflict):
        orch.trigger("manual", background=False)
    # a restart marks the orphan as failed, which unblocks new runs
    Orchestrator(s, st.store, st.registry)
    assert st.store.get_run("run-fake")["status"] == "failed"


def test_minimum_data_check_skips(env):
    s, _ = env
    st, orch = make_orch(s.with_overrides(retrain_min_fraud=10**6))
    run = st.store.get_run(orch.trigger("manual", background=False)["id"])
    assert run["status"] == "skipped" and st.registry.active() == env[1]


def test_failure_is_recorded_and_old_model_kept(env, tmp_path):
    s, v0 = env
    st, orch = make_orch(s.with_overrides(data_path=tmp_path / "missing.csv"))
    run = st.store.get_run(orch.trigger("manual", background=False)["id"])
    assert run["status"] == "failed" and "Dataset not found" in run["error"] and st.registry.active() == v0


def test_invalid_overrides_rejected(env):
    _, orch = make_orch(env[0])
    with pytest.raises(ValueError):
        orch.trigger("manual", overrides={"evil": 1})


def test_drift_triggered_path_requires_drift_report(env, client):
    s, _ = env
    st, orch = make_orch(s)
    with pytest.raises(RetrainingRejected):
        orch.trigger("drift", drift_report_id="nope", background=False)


def test_rollback_and_registry_rules(env):
    s, v0 = env
    st, orch = make_orch(s)
    with pytest.raises(Exception, match="no previously approved"):
        orch.rollback("x")
    run = st.store.get_run(orch.trigger("manual", overrides={"n_estimators": 40, "seed": 3, "bootstrap": 5},
                                        background=False)["id"])
    assert run["status"] == "promoted"
    back = orch.rollback("test rollback")
    assert back == v0 and st.predictor.version == v0 and st.registry.active() == v0
    assert st.registry.snapshot()["versions"][run["candidate_version"]]["status"] == "rolled_back"


def test_end_to_end_via_api_drift_triggers_retraining(env):
    from fastapi.testclient import TestClient

    from app.api.main import create_app
    with TestClient(create_app(env[0])) as c:
        sim = c.post("/monitor/simulate", json={"n": 300, "magnitude": 3.0, "auto_trigger": True}).json()
        assert sim["retraining"]["triggered"] is True
        c.app.state.s.orchestrator.wait_idle()
        runs = c.get("/retraining/status").json()["runs"]
        assert runs[0]["trigger"] == "drift" and runs[0]["status"] in {"promoted", "rejected"}
        assert runs[0]["drift_report_id"] == sim["after"]["id"]


def test_api_manual_trigger_conflict_and_rollback_endpoint(env):
    from fastapi.testclient import TestClient

    from app.api.main import create_app
    with TestClient(create_app(env[0])) as c:
        r = c.post("/retraining/trigger", json={"reason": "t", "overrides": {"n_estimators": 40, "bootstrap": 5}})
        assert r.status_code == 202
        c.app.state.s.orchestrator.wait_idle()
        assert c.get("/retraining/status").json()["runs"][0]["status"] in {"promoted", "rejected"}
        assert c.post("/retraining/trigger", json={"overrides": {"bogus": 1}}).status_code == 422
        if c.get("/model/versions").json()["active"] != env[1]:
            assert c.post("/retraining/rollback", json={}).status_code == 200
            assert c.get("/health").json()["model_version"] == env[1]
        else:
            assert c.post("/retraining/rollback", json={}).status_code == 422
