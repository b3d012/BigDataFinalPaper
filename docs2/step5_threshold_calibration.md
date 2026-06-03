# Step 5 Threshold Calibration

## What was added

- Added a dedicated threshold calibration script: `src/edge_iiot_thresholds.py`.
- The script reads the saved holdout probabilities and, when available, the out-of-fold CV probabilities from the SMOTE + CV path.
- It evaluates a fixed grid of thresholds and saves paper-friendly comparison outputs.

## How thresholds are evaluated

- Thresholds are swept from `0.05` to `0.95` with a fixed step of `0.01` by default.
- For each threshold, the script computes:
  - accuracy
  - attack precision
  - attack recall
  - attack FNR
  - TP, FP, TN, FN
  - F1 and F2 for tradeoff analysis
- PR-AUC is reported as score context from the raw probabilities, because it does not change with threshold.
- The evaluation is deterministic because it uses saved predictions and a fixed threshold grid.

## How recommended thresholds are selected

- `max F1`: threshold with the highest F1 score.
- `max F2`: threshold with the highest F2 score, which emphasizes recall more strongly.
- `highest recall under minimum precision`: best attack recall among thresholds meeting the precision constraint.
- `lowest FNR under minimum precision`: best missed-attack rate among thresholds meeting the precision constraint.
- The default minimum precision constraint is `0.97`, which keeps the threshold search aligned with a paper-oriented false-alarm tolerance.

## Outputs produced

- `output/reports/edge_iiot_holdout_threshold_grid.csv`
- `output/reports/edge_iiot_holdout_threshold_recommendations.json`
- `output/reports/edge_iiot_cv_threshold_grid.csv`
- `output/reports/edge_iiot_cv_threshold_recommendations.json`
- `output/reports/edge_iiot_threshold_calibration_summary.md`
- `output/figures/edge_iiot_holdout_precision_recall_vs_threshold.png`
- `output/figures/edge_iiot_holdout_fnr_vs_threshold.png`
- `output/figures/edge_iiot_cv_precision_recall_vs_threshold.png`
- `output/figures/edge_iiot_cv_fnr_vs_threshold.png`

## How this supports the paper methodology

- Threshold calibration makes the attack-detection tradeoff explicit instead of relying on a single default cutoff.
- The paper can discuss recall/FNR gains separately from precision loss.
- Because the calibration uses saved probabilities, it does not alter the classifier or preprocessing pipeline.
- Holdout and CV calibration give two complementary views:
  - holdout for the original offline baseline
  - out-of-fold CV for the SMOTE-balanced variant
