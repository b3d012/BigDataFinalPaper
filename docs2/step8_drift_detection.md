# Step 8 Drift Detection

## What was added

- Added an offline drift detection script at `src/edge_iiot_drift.py`.
- The drift layer is separate from the classifier and the anomaly detector.
- It compares reference and target batches in the transformed feature space produced by the saved classifier feature contract.
- It uses PSI as the primary drift score and also computes Jensen-Shannon divergence as a supporting distance measure.

## How drift is measured

- The script first reconstructs the classifier feature contract from `models/edge_iiot_xgb_model.joblib`.
- Raw batches are normalized and converted into the same typed feature frame used by the classifier rebuild.
- A reference preprocessing pipeline is fit on the reference batch only.
- The reference and target batches are transformed into a shared feature space.
- For each transformed feature, the script computes:
  - Population Stability Index
  - Jensen-Shannon divergence
  - reference and target means
  - mean shift
  - reference and target standard deviations

## Supported comparisons

- `dataset` mode compares the classifier train split against the classifier holdout split.
- `demo` mode compares benign reference traffic from the dataset against the demo replay rows.
- The demo mode prefers row-level demo replay output. If only the row-level extracted demo CSVs are available, they can also be used.

## Files produced

- `output/reports/edge_iiot_drift_feature_scores.csv`
- `output/reports/edge_iiot_drift_summary.json`
- `output/reports/edge_iiot_drift_run_summary.md`
- `output/figures/edge_iiot_drift_top_features.png`
- `output/demo/edge_iiot_demo_drift_summary.json`

## How this supports the paper methodology

- The drift layer gives a simple, explainable way to compare older reference traffic against newer traffic batches.
- PSI is easy to summarize in the paper and produces a per-feature ranking that highlights which signals shifted the most.
- The overall severity rule is transparent:
  - low if max PSI is below `0.10`
  - moderate if max PSI is between `0.10` and `0.25`
  - high if max PSI is at least `0.25`

## Why retraining was not added yet

- This step is only about detecting drift, not adapting the model.
- Retaining the current trained classifier keeps the paper narrative clean: first detect drift, then discuss how a future system might trigger retraining.
- Adaptive retraining is intentionally deferred to a later step so the paper can separate detection from response.
