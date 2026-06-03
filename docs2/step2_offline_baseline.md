# Step 2 Offline Baseline

## What was implemented

- Rebuilt the root `src/edge_iiot_experiment.py` as an offline-only Edge-IIoT trainer/evaluator.
- Preserved the archived working pipeline behavior:
  - column normalization
  - binary label construction from `Attack_label` or `Attack_type`
  - numeric versus categorical typing
  - default dropping of identity and payload-heavy columns
  - median imputation for numeric features
  - missing-fill plus one-hot encoding for categorical features
  - stratified train/validation/test splitting
  - threshold selection with `fixed`, `f1`, and `f2` options
  - final full-data model training after holdout evaluation
- Added paper-friendly offline reporting:
  - accuracy
  - ROC-AUC
  - PR-AUC
  - confusion matrix
  - per-class precision/recall via classification report
  - attack FNR derived from the confusion matrix
- Saved the working artifacts in the current repo structure.

## What was intentionally left out

- Live capture and monitor loops
- tshark, PCAP extraction, and PCAP-folder scoring
- compare/reporting for local PCAP schema matching
- Spark, MongoDB, dashboards, and streaming deployment code
- SHAP explainability
- drift detection or adaptive retraining
- anomaly detection layers
- SMOTE or other class balancing before the split

## Files produced

- `models/edge_iiot_xgb_model.joblib`
- `models/edge_iiot_xgb_model.metadata.json`
- `models/edge_iiot_xgb_model.feature_importance.csv`
- `output/reports/edge_iiot_holdout_metrics.json`
- `output/reports/edge_iiot_holdout_predictions.csv`
- `output/reports/edge_iiot_holdout_confusion_matrix.csv`
- `output/reports/edge_iiot_classification_report.csv`

## Next recommended step

- Add class balancing only on the training split, or an equivalent safe imbalance strategy, and compare it against this baseline before changing the feature or model family.
- After that, add paper-facing threshold calibration and class-specific reporting.
