"""Evaluation metrics, threshold selection and bootstrap uncertainty."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score


def compute_metrics(y, score, thr: float) -> dict:
    y = np.asarray(y)
    score = np.asarray(score)
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(thr), "n": int(len(y)), "n_fraud": int(y.sum()),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "pr_auc": float(average_precision_score(y, score)) if both else None,
        "roc_auc": float(roc_auc_score(y, score)) if both else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_positive_count": int(fp),
    }


def select_threshold(y, score, target_recall: float) -> tuple[float, dict]:
    """Objective: among thresholds with validation recall >= target_recall, pick the one with the highest
    precision (ties -> higher threshold). If no threshold reaches the target, fall back to max-F1."""
    prec, rec, thr = precision_recall_curve(y, score)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= target_recall
    if ok.any():
        idx = np.where(ok)[0]
        best = idx[np.lexsort((thr[idx], prec[idx]))[-1]]
        return float(thr[best]), {"objective": f"max precision s.t. recall >= {target_recall}", "fallback": False}
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best = int(np.argmax(f1))
    return float(thr[best]), {"objective": "max F1 (target recall unreachable)", "fallback": True}


def pr_curve_points(y, score, max_points: int = 150) -> dict:
    """One point per distinct recall level (its best precision), in recall order: the exact curve shape in few points."""
    prec, rec, _ = precision_recall_curve(y, score)
    best: dict[float, float] = {}
    for p, r in zip(prec, rec, strict=True):
        best[r] = max(p, best.get(r, 0.0))
    rec = np.array(sorted(best))
    prec = np.array([best[r] for r in rec])
    if len(rec) > max_points:
        idx = np.unique(np.linspace(0, len(rec) - 1, max_points).astype(int))
        prec, rec = prec[idx], rec[idx]
    return {"precision": prec.tolist(), "recall": rec.tolist()}


def bootstrap_ci(y, score, thr: float, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap (row resampling) 95% CIs. Wide intervals = too few fraud cases for precise claims."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {"pr_auc": [], "precision": [], "recall": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        m = compute_metrics(y[i], score[i], thr)
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))]
            for k, v in out.items() if v} | {"n_boot": n_boot}



def value_metrics(y, score, thr: float, amount, review_cost: float) -> dict:
    """Money view of one operating point. Cost = fraud value missed + review_cost per flagged transaction."""
    y, amount = np.asarray(y), np.asarray(amount, dtype=float)
    flag = np.asarray(score) >= thr
    fraud_total = float(amount[y == 1].sum())
    caught = float(amount[(y == 1) & flag].sum())
    return {"flagged": int(flag.sum()), "fraud_value_total": fraud_total, "fraud_value_caught": caught,
            "fraud_value_caught_share": caught / fraud_total if fraud_total else 0.0,
            "fraud_value_missed": fraud_total - caught, "legit_value_flagged": float(amount[(y == 0) & flag].sum()),
            "review_cost": review_cost, "total_cost": fraud_total - caught + review_cost * int(flag.sum())}
