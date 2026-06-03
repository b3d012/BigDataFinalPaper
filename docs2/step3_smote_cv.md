# Step 3 SMOTE + CV

## What was added

- Added an optional SMOTE + stratified cross-validation training mode to `src/edge_iiot_experiment.py`.
- Kept the baseline holdout training path intact.
- Added CLI flags:
  - `--use_smote`
  - `--cv_folds 5`
- Added CV report outputs under `output/reports/`.

## How SMOTE is applied safely

- SMOTE is applied only to the training portion of each CV fold.
- The preprocessing pipeline is fit on the fold training split only.
- The validation split remains untouched.
- The fold validation predictions are scored on the original validation data, not on resampled data.
- For SMOTE folds, `scale_pos_weight` is neutralized to `1.0` to avoid double-compensating for imbalance.

## How CV is performed

- Uses stratified `KFold` splitting.
- Default is 5 folds.
- For each fold:
  - fit preprocessing on the fold training split
  - transform train and validation splits
  - apply SMOTE only to transformed training data
  - train XGBoost on the resampled training fold
  - evaluate the untouched validation fold at a fixed threshold of `0.5`
- Aggregates:
  - per-fold accuracy
  - per-fold ROC-AUC
  - per-fold PR-AUC
  - per-fold confusion matrix
  - per-fold attack recall
  - per-fold attack FNR
  - mean and standard deviation across folds

## Files produced

- `models/edge_iiot_xgb_model.joblib`
- `models/edge_iiot_xgb_model.metadata.json`
- `models/edge_iiot_xgb_model.feature_importance.csv`
- `output/reports/edge_iiot_holdout_metrics.json`
- `output/reports/edge_iiot_holdout_predictions.csv`
- `output/reports/edge_iiot_holdout_confusion_matrix.csv`
- `output/reports/edge_iiot_classification_report.csv`
- `output/reports/edge_iiot_cv_fold_metrics.csv`
- `output/reports/edge_iiot_cv_predictions.csv`
- `output/reports/edge_iiot_cv_confusion_matrix.csv`
- `output/reports/edge_iiot_cv_summary.json`

## What still remains

- SHAP explainability
- per-class threshold calibration beyond the fixed-threshold CV pass
- drift detection and adaptive retraining
- anomaly detection layer
- live capture, PCAP monitoring, and deployment code
- multi-class expansion, if needed later for the paper
