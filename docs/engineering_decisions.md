# Engineering decisions

| Decision | Why | Trade-off |
|---|---|---|
| Kaggle credit-card dataset | Public, labelled, extreme imbalance, has `Time` for chronological split | PCA features not interpretable; 2 days of data |
| Chronological split | Mirrors deployment (train on past, test on future); random splits leak temporal structure | Fewer frauds per split |
| PR-AUC primary, fixed-recall comparison | Accuracy is meaningless at 0.17 % positives; fixed recall makes FP counts comparable | Operating point is an assumption (target recall 0.80) |
| Threshold tuned on validation only | Test remains an unbiased final estimate | Threshold may be noisy with few validation frauds |
| Ensemble must earn its place | Unsupervised scores can add noise | Extra code that may be unused |
| Evidently 0.4.40 legacy API | Stable, verified here; newer releases changed the API | Pinned to an older line; numpy<2.1 |
| SQLite + single background thread | Zero infrastructure, easy to test and demo | Not horizontally scalable; one uvicorn worker only |
| Retrain on historical labels, not drift events | Unlabelled/simulated data cannot supervise training | Retraining is a workflow demo, not a real feedback loop |
| Gate compares on the same validation split | Apples-to-apples comparison with the incumbent | Validation is also used for early stopping/thresholds (documented) |
| Checksum-verified joblib | Detect corruption; pickle risk documented | Not a defence against a malicious artifact author |
| Dashboard is static HTML/JS served by the API, talking to it over HTTP | One container and port; proves the API is real | No server-side rendering |
