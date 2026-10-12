"""Write the human-readable evaluation report for the active model from its stored metrics.json (no new numbers)."""
import json
from pathlib import Path

import _common  # noqa: F401

from app.common.config import Settings
from app.models.artifacts import read_manifest
from app.registry.registry import ModelRegistry

s = Settings.from_env()
reg = ModelRegistry(s.models_dir)
v = reg.active()
if not v:
    raise SystemExit("No active model. Run scripts/train.py first.")
path = reg.artifact_path(v)
man, rep = read_manifest(path), json.loads((path / "metrics.json").read_text())
f = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
L = [f"# Model evaluation - {v}", "",
     f"- Dataset: `{man['dataset'].get('path')}` - mode: **{man['dataset'].get('mode')}**",
     f"- Split: chronological 60/20/20 (train/val/test): `{json.dumps(man['splits'])}`",
     f"- Threshold objective: {rep['models']['xgboost']['threshold_selection']['objective']} (chosen on VALIDATION); "
     "all numbers below marked *test* use the held-out test split exactly once.",
     f"- Decision model in production path: **{rep['decision_model']}** (threshold {rep['threshold']:.4f})", "",
     "| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP | FPR | Threshold |", "|---|---|---|---|---|---|---|---|---|---|"]
for n, m in rep["models"].items():
    for sp in ("validation", "test"):
        x = m[sp]
        L.append(f"| {n} | {sp} | {f(x['precision'])} | {f(x['recall'])} | {f(x['f1'])} | {f(x['pr_auc'])} | {f(x['roc_auc'])} | "
                 f"{x['false_positive_count']} | {x['false_positive_rate']:.5f} | {x['threshold']:.4f} |")
# FP reduction of the serving model vs. the baseline; both thresholds come from the same validation recall policy
dm = rep["decision_model"]
fpr: dict = {}
L += ["", f"## False-positive reduction: {dm} vs. baseline", "",
      f"Both thresholds chosen on VALIDATION for recall >= {rep['target_recall']}; the test split is then scored once.", "",
      f"| Split | Baseline FP | {dm} FP | FP reduction | Baseline recall (TP) | {dm} recall (TP) |", "|---|---|---|---|---|---|"]
for sp in ("validation", "test"):
    b, d = rep["models"]["baseline"][sp], rep["models"][dm][sp]
    red = 1 - d["fp"] / b["fp"] if b["fp"] else None
    fpr[sp] = {"baseline_fp": b["fp"], "model_fp": d["fp"], "reduction": red, "baseline_tp": b["tp"], "model_tp": d["tp"]}
    L.append(f"| {sp} | {b['fp']} | {d['fp']} | {'n/a' if red is None else f'{red:.1%}'} | "
             f"{b['recall']:.3f} ({b['tp']}) | {d['recall']:.3f} ({d['tp']}) |")
rep["fp_reduction_vs_baseline"] = {"decision_model": dm, **fpr}

# money view of the same operating points on the test split (computed at training time)
if "test_value" not in rep["models"][dm]:
    raise SystemExit("This model predates the money metrics; retrain it with scripts/train.py.")
val = {n: rep["models"][n]["test_value"] for n in ("baseline", dm)}
b, d = val["baseline"], val[dm]
eur = lambda x: f"EUR {x:,.0f}"  # noqa: E731
pct = lambda new, old: f"{(new - old) / old:+.0%}" if old else "n/a"  # noqa: E731
L += ["", "## Business impact (test split, EUR)", "",
      f"Same thresholds as above. Total cost = fraud value missed + EUR {d['review_cost']:g} per transaction flagged for review.", "",
      f"| | Baseline | {dm} | Change |", "|---|---|---|---|",
      f"| Transactions flagged for review | {b['flagged']} | {d['flagged']} | {pct(d['flagged'], b['flagged'])} |",
      f"| Fraud value caught | {eur(b['fraud_value_caught'])} ({b['fraud_value_caught_share']:.0%}) | "
      f"{eur(d['fraud_value_caught'])} ({d['fraud_value_caught_share']:.0%}) | |",
      f"| Legitimate spend flagged | {eur(b['legit_value_flagged'])} | {eur(d['legit_value_flagged'])} | "
      f"{pct(d['legit_value_flagged'], b['legit_value_flagged'])} |",
      f"| Total cost | {eur(b['total_cost'])} | {eur(d['total_cost'])} | {pct(d['total_cost'], b['total_cost'])} |",
      f"| Total cost without screening (all fraud lost) | {eur(d['fraud_value_total'])} | {eur(d['fraud_value_total'])} | |"]
g = lambda iv: f"[{iv[0]:.3f}, {iv[1]:.3f}]"  # noqa: E731
L += ["", "## 95% bootstrap intervals (test)", "", "| Model | PR-AUC | Precision | Recall |", "|---|---|---|---|"]
for n, m in rep["models"].items():
    c = m["test_ci95"]
    L.append(f"| {n} | {g(c['pr_auc'])} | {g(c['precision'])} | {g(c['recall'])} |")
e = rep["ensemble_analysis"]
best = max([("xgboost", e["validation_pr_auc_xgboost"])] +
           [(f"ensemble w={w}", ap) for w, ap in e["validation_pr_auc_by_weight"].items()], key=lambda x: x[1])
L += ["", "## Automated model selection (validation)", "",
      f"- Candidates: XGBoost alone and XGBoost + Isolation Forest blends (weights {list(e['validation_pr_auc_by_weight'])})",
      f"- Validation PR-AUC: XGBoost {e['validation_pr_auc_xgboost']:.4f}; blends `{e['validation_pr_auc_by_weight']}`",
      f"- Selected for serving: **{rep['decision_model']}** (best validation PR-AUC: {best[0]}, {best[1]:.4f})"]
out = Path("reports/model_evaluation_report.md")
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(L))
(out.parent / "model_evaluation_report.json").write_text(json.dumps(rep, indent=1))
print("\n".join(L)); print(f"\nwritten: {out}")
