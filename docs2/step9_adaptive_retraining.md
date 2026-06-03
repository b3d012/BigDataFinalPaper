# Step 9 Adaptive Retraining

This step adds an offline, drift-triggered retraining workflow for the Edge-IIoT paper pipeline. It does not replace the existing XGBoost baseline, SMOTE + CV, SHAP, threshold calibration, demo replay, anomaly detection, or drift detection layers.

## What Was Added

- A dedicated retraining script: `src/edge_iiot_retrain.py`
- A retrained model bundle saved separately from the original classifier
- A before/after comparison report for the original versus retrained model
- A trigger report that records why retraining was or was not activated

The retraining workflow reuses the same binary XGBoost family and the same preprocessing contract as the main classifier.

## Retraining Trigger Rule

Retraining is not always on. It is triggered from the existing drift layer output using a simple rule:

- retrain if drift severity is `high`
- retrain if drift severity is `moderate` and at least 12 transformed features have PSI `>= 0.10`

In the current implementation, `high` severity is defined by `max PSI >= 0.25`, while `moderate` starts at `PSI >= 0.10`.

This makes the trigger easy to explain in the paper and keeps the decision tied to measurable distribution shift.

## How the Retraining Data Is Constructed

The workflow uses a deterministic contiguous split of the deduplicated dataset:

- first 60 percent: historical reference batch
- next 20 percent: drift / retraining batch
- final 20 percent: fixed evaluation target

The retraining model is fit on the reference batch plus the drift batch. The evaluation target is held out and used for both the original model and the retrained model so the comparison stays fair.

This is an offline simulation of adaptive retraining. It is intentionally simple so the paper can describe the adaptation logic clearly.

## Outputs Produced

The workflow writes:

- `models/edge_iiot_xgb_model_retrained.joblib`
- `models/edge_iiot_xgb_model_retrained.metadata.json`
- `output/reports/edge_iiot_retrain_trigger.json`
- `output/reports/edge_iiot_retrain_comparison.json`
- `output/reports/edge_iiot_retrain_comparison.csv`
- `output/reports/edge_iiot_retrain_run_summary.md`

The trigger report records:

- whether retraining was activated
- the trigger reason
- trigger severity
- reference, drift, and evaluation row counts
- the top drifted features

The comparison report records:

- accuracy
- ROC-AUC
- PR-AUC
- attack precision
- attack recall
- attack FNR
- metric deltas between the original and retrained model

## How Performance Is Compared

The original saved classifier bundle is evaluated on the fixed evaluation target first. If retraining is triggered, the retrained bundle is evaluated on the same target.

The comparison is reported as:

- original metric value
- retrained metric value
- delta

The summary also classifies the outcome as:

- `helped`
- `hurt`
- `little_difference`

This keeps the before/after result easy to interpret for the paper.

## Limitations

- This is an offline simulation, not a production retraining system.
- The batch split is contiguous and synthetic, so it approximates temporal drift rather than proving it in a live environment.
- Retraining is driven by drift statistics, not by live feedback loops.
- The current run showed a slight performance decrease after retraining, which is useful to report as a cautionary result rather than a failure.

## Paper Relevance

This step supports the methodology section by showing how drift can trigger controlled adaptation, how a retrained model can be compared against the original model on a fixed target, and how retraining decisions can be made transparent and reproducible.
