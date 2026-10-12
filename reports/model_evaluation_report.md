# Model evaluation - v20261012085427-adf95d2a

- Dataset: `data/raw/creditcard.csv` - mode: **full**
- Split: chronological 60/20/20 (train/val/test): `{"train": {"rows": 170235, "fraud": 342, "fraud_rate": 0.002008987575997885, "time_min": 0.0, "time_max": 120392.0}, "val": {"rows": 56745, "fraud": 57, "fraud_rate": 0.0010044937879989425, "time_min": 120393.0, "time_max": 145233.0}, "test": {"rows": 56746, "fraud": 74, "fraud_rate": 0.0013040566735981391, "time_min": 145234.0, "time_max": 172792.0}}`
- Threshold objective: max precision s.t. recall >= 0.8 (chosen on VALIDATION); all numbers below marked *test* use the held-out test split exactly once.
- Decision model in production path: **xgboost** (threshold 0.5186)

| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP | FPR | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| baseline | validation | 0.2629 | 0.8070 | 0.3966 | 0.7532 | 0.9691 | 129 | 0.00228 | 0.9651 |
| baseline | test | 0.4052 | 0.8378 | 0.5463 | 0.7442 | 0.9815 | 91 | 0.00161 | 0.9651 |
| xgboost | validation | 0.4107 | 0.8070 | 0.5444 | 0.7753 | 0.9795 | 66 | 0.00116 | 0.5186 |
| xgboost | test | 0.5377 | 0.7703 | 0.6333 | 0.7762 | 0.9766 | 49 | 0.00086 | 0.5186 |
| iforest | validation | 0.0129 | 0.8246 | 0.0254 | 0.0196 | 0.9344 | 3603 | 0.06356 | 0.9473 |
| iforest | test | 0.0188 | 0.7568 | 0.0367 | 0.0323 | 0.9414 | 2922 | 0.05156 | 0.9473 |
| ensemble | validation | 0.3866 | 0.8070 | 0.5227 | 0.7747 | 0.9636 | 73 | 0.00129 | 0.5941 |
| ensemble | test | 0.5229 | 0.7703 | 0.6230 | 0.7744 | 0.9668 | 52 | 0.00092 | 0.5941 |

## False-positive reduction: xgboost vs. baseline

Both thresholds chosen on VALIDATION for recall >= 0.8; the test split is then scored once.

| Split | Baseline FP | xgboost FP | FP reduction | Baseline recall (TP) | xgboost recall (TP) |
|---|---|---|---|---|---|
| validation | 129 | 66 | 48.8% | 0.807 (46) | 0.807 (46) |
| test | 91 | 49 | 46.2% | 0.838 (62) | 0.770 (57) |

## Business impact (test split, EUR)

Same thresholds as above. Total cost = fraud value missed + EUR 5 per transaction flagged for review.

| | Baseline | xgboost | Change |
|---|---|---|---|
| Transactions flagged for review | 153 | 106 | -31% |
| Fraud value caught | EUR 5,357 (69%) | EUR 5,327 (69%) | |
| Legitimate spend flagged | EUR 66,282 | EUR 35,420 | -47% |
| Total cost | EUR 3,135 | EUR 2,931 | -7% |
| Total cost without screening (all fraud lost) | EUR 7,728 | EUR 7,728 | |

## 95% bootstrap intervals (test)

| Model | PR-AUC | Precision | Recall |
|---|---|---|---|
| baseline | [0.626, 0.846] | [0.338, 0.481] | [0.753, 0.908] |
| xgboost | [0.682, 0.863] | [0.444, 0.625] | [0.666, 0.861] |
| iforest | [0.022, 0.048] | [0.015, 0.023] | [0.663, 0.861] |
| ensemble | [0.683, 0.863] | [0.431, 0.609] | [0.666, 0.861] |

## Automated model selection (validation)

- Candidates: XGBoost alone and XGBoost + Isolation Forest blends (weights ['0.9', '0.8', '0.7', '0.5'])
- Validation PR-AUC: XGBoost 0.7753; blends `{'0.9': 0.773515594148025, '0.8': 0.7746551190774611, '0.7': 0.7732901512035532, '0.5': 0.768001878756087}`
- Selected for serving: **xgboost** (best validation PR-AUC: xgboost, 0.7753)