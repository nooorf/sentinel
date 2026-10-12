"""Train + compare baseline, XGBoost, Isolation Forest and a validated ensemble.

Protocol (no test-set leakage):
  * preprocessing, all models and the IF score normaliser are fitted on TRAIN only
  * XGBoost early-stops on VALIDATION; thresholds, ensemble weight and ensemble inclusion are chosen on VALIDATION
  * TEST is used once, for reporting
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from app.data.schema import FEATURES, LABEL, Splits
from app.features.preprocess import build_preprocessor
from app.models.metrics import bootstrap_ci, compute_metrics, pr_curve_points, select_threshold, value_metrics
from app.models.scoring import score_parts

ALLOWED_OVERRIDES = {"n_estimators", "max_depth", "learning_rate", "seed", "target_recall", "bootstrap"}


@dataclass
class TrainConfig:
    seed: int = 42
    target_recall: float = 0.80
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    early_stopping_rounds: int = 30
    if_n_estimators: int = 200
    ensemble_weights: list[float] = field(default_factory=lambda: [0.9, 0.8, 0.7, 0.5])
    ensemble_min_gain: float = 0.005  # min validation PR-AUC gain over XGBoost alone to keep the ensemble
    bootstrap: int = 200
    review_cost: float = 5.0  # assumed EUR cost of manually reviewing one flagged transaction (money view only)

    def with_overrides(self, overrides: dict | None) -> TrainConfig:
        overrides = overrides or {}
        bad = set(overrides) - ALLOWED_OVERRIDES
        if bad:
            raise ValueError(f"Unsupported overrides: {sorted(bad)}; allowed: {sorted(ALLOWED_OVERRIDES)}")
        d = asdict(self)
        d.update(overrides)
        return TrainConfig(**d)


def train_pipeline(splits: Splits, cfg: TrainConfig) -> dict:
    tr, va, te = splits.train, splits.val, splits.test
    ytr, yva, yte = tr[LABEL].to_numpy(), va[LABEL].to_numpy(), te[LABEL].to_numpy()

    pre = build_preprocessor().fit(tr[FEATURES])  # fitted on train only
    Ztr = pre.transform(tr[FEATURES])
    Zva = pre.transform(va[FEATURES])

    baseline = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=cfg.seed).fit(Ztr, ytr)

    spw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))  # class weighting; no resampling of any split
    xgb = XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
        early_stopping_rounds=cfg.early_stopping_rounds, tree_method="hist", random_state=cfg.seed, n_jobs=2)
    xgb.fit(Ztr, ytr, eval_set=[(Zva, yva)], verbose=False)

    iforest = IsolationForest(n_estimators=cfg.if_n_estimators, random_state=cfg.seed, n_jobs=2).fit(Ztr)  # no labels
    raw_tr = -iforest.score_samples(Ztr)
    quantiles = np.quantile(raw_tr, np.linspace(0, 1, 1001))

    bundle = {"preprocessor": pre, "baseline": baseline, "xgb": xgb, "iforest": iforest,
              "if_quantiles": quantiles, "ensemble_weight": 1.0, "decision_model": "xgboost",
              "threshold": 0.5, "features": list(FEATURES)}
    sv = score_parts(bundle, va)

    # ---- ensemble weight search on VALIDATION PR-AUC (w=1 means XGBoost alone)
    ap_xgb = average_precision_score(yva, sv["xgboost"])
    grid = {}
    for w in cfg.ensemble_weights:
        grid[w] = float(average_precision_score(yva, w * sv["xgboost"] + (1 - w) * sv["iforest"]))
    best_w = max(grid, key=grid.get)
    gain = grid[best_w] - ap_xgb
    keep_ensemble = gain >= cfg.ensemble_min_gain
    bundle["ensemble_weight"] = float(best_w) if keep_ensemble else 1.0
    bundle["decision_model"] = "ensemble" if keep_ensemble else "xgboost"
    # scoring again with final weight so "ensemble" column reflects the best weight found
    best_ens_bundle = {**bundle, "ensemble_weight": float(best_w)}
    sv = score_parts(best_ens_bundle, va)
    st = score_parts(best_ens_bundle, te)

    # ---- per-model thresholds on VALIDATION; final metrics on TEST
    report: dict = {"models": {}}
    thresholds = {}
    for name in ("baseline", "xgboost", "iforest", "ensemble"):
        thr, tinfo = select_threshold(yva, sv[name], cfg.target_recall)
        thresholds[name] = thr
        report["models"][name] = {
            "threshold": thr, "threshold_selection": tinfo,
            "validation": compute_metrics(yva, sv[name], thr),
            "test": compute_metrics(yte, st[name], thr),
            "test_ci95": bootstrap_ci(yte, st[name], thr, n_boot=cfg.bootstrap, seed=cfg.seed),
            "pr_curve_test": pr_curve_points(yte, st[name]),
            "test_value": value_metrics(yte, st[name], thr, te["Amount"].to_numpy(), cfg.review_cost),
        }
    bundle["threshold"] = float(thresholds["ensemble" if keep_ensemble else "xgboost"])

    # latency (model-only, single row, warm) for the decision path
    one = te.iloc[[0]]
    for _ in range(10):
        score_parts(bundle, one)
    t = []
    for i in range(100):
        s = time.perf_counter()
        score_parts(bundle, te.iloc[[i % len(te)]])
        t.append((time.perf_counter() - s) * 1000)
    report["model_only_latency_ms_single"] = {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
                                              "n": len(t), "note": "indicative; use scripts/benchmark.py for the full benchmark"}

    report["ensemble_analysis"] = {
        "validation_pr_auc_xgboost": float(ap_xgb), "validation_pr_auc_by_weight": grid,
        "best_weight": float(best_w), "gain_over_xgboost": float(gain),
        "min_gain_required": cfg.ensemble_min_gain, "ensemble_selected": bool(keep_ensemble),
        "conclusion": ("Ensemble kept: validation PR-AUC gain >= minimum." if keep_ensemble else
                       "Ensemble NOT selected: gain over XGBoost alone is below the minimum; simpler model kept."),
    }
    report["decision_model"] = bundle["decision_model"]
    report["threshold"] = bundle["threshold"]
    report["best_iteration"] = int(getattr(xgb, "best_iteration", cfg.n_estimators) or 0)
    report["splits"] = splits.info
    report["target_recall"] = cfg.target_recall
    bundle["report"] = report
    bundle["config"] = asdict(cfg)
    return bundle
