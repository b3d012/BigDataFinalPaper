# Edge-IIoT Anomaly Layer Summary

- Classifier bundle: models\edge_iiot_xgb_model.joblib
- Anomaly bundle: models\edge_iiot_isolation_forest.joblib
- Threshold strategy: benign_score_quantile_0.0500
- Decision threshold: -0.241170
- Contamination: 0.0500
- Benign score quantile: 0.0500
- Dense fallback used: False

## Holdout Metrics
- ROC-AUC: 0.5240
- PR-AUC: 0.8384
- Attack precision: 0.8497
- Attack recall: 0.9807
- Attack FNR: 0.0193

## Score Summary
- train_benign_anomaly_score: count=19441 mean=-0.144614 std=0.079860
- holdout_anomaly_score: count=31398 mean=-0.138297 std=0.071729

## Demo Replay
- Files scored: 15
- Demo attack file predictions: 6
- Demo benign file predictions: 9
