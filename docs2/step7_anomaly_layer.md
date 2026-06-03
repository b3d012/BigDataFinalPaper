# Step 7 Anomaly Layer

## What was added

- Added an offline anomaly detection script at `src/edge_iiot_anomaly.py`.
- The anomaly layer is separate from the XGBoost classifier and does not modify the classifier pipeline.
- It trains an Isolation Forest on benign training rows only, using the same feature contract as the saved classifier bundle.
- It evaluates the anomaly detector on the held-out split and optionally scores the demo replay output when available.

## How the anomaly detector is trained

- The script loads `models/edge_iiot_xgb_model.joblib` and reuses its saved preprocessing contract.
- The Edge-IIoT dataset is prepared with the same column normalization, label construction, and identity/payload filtering used by the classifier rebuild.
- The feature preprocessor is fit on the training split only.
- Isolation Forest is then fit only on the benign subset of the training split.
- The anomaly threshold is calibrated from the benign training score distribution using a configurable benign-score quantile.

## What data it uses

- Primary training data: `data/ML-EdgeIIoT-dataset.csv`
- Saved classifier bundle: `models/edge_iiot_xgb_model.joblib`
- Optional demo input: `output/demo/edge_iiot_demo_predictions.csv`

## Files produced

- `models/edge_iiot_isolation_forest.joblib`
- `models/edge_iiot_isolation_forest.metadata.json`
- `output/reports/edge_iiot_anomaly_holdout_metrics.json`
- `output/reports/edge_iiot_anomaly_holdout_predictions.csv`
- `output/reports/edge_iiot_anomaly_confusion_matrix.csv`
- `output/reports/edge_iiot_anomaly_classifier_comparison.csv`
- `output/reports/edge_iiot_anomaly_classification_report.csv`
- `output/reports/edge_iiot_anomaly_run_summary.md`
- `output/demo/edge_iiot_demo_anomaly_predictions.csv`
- `output/demo/edge_iiot_demo_anomaly_summary.csv`
- `output/demo/edge_iiot_demo_anomaly_vs_classifier.csv`

## How it complements the classifier

- The XGBoost classifier remains the primary supervised detector.
- The anomaly layer adds a second offline signal that is trained only on benign traffic.
- The two outputs can be compared on the held-out split to see where they agree or diverge.
- This is useful for the paper discussion because the anomaly layer can be described as a complementary detector rather than a replacement model.

## Limitations

- Isolation Forest is still weaker than the supervised classifier on score separation, so the anomaly ROC-AUC is much lower than the classifier ROC-AUC.
- The anomaly threshold is a paper-friendly calibration choice based on benign training scores, not a fully optimized operating point.
- Demo file-level aggregation uses clear offline defaults and should be treated as a descriptive summary, not a deployment rule.
- The anomaly layer remains offline-only and intentionally excludes live capture, streaming, dashboards, Spark, MongoDB, and drift handling.
