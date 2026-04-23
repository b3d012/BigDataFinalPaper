# Step 4 SHAP Explainability

## What was added

- Added a dedicated SHAP analysis script: `src/edge_iiot_shap.py`.
- The script loads the saved model bundle from `models/edge_iiot_xgb_model.joblib`.
- It reconstructs the same holdout split used by the offline baseline so the explanations are tied to the trained model and the same feature contract.
- It produces both global and local SHAP outputs without changing the classifier itself.

## How SHAP is computed

- The script uses the trained XGBoost booster from the saved bundle.
- The saved preprocessor transforms the holdout input into the exact model feature space.
- Global importance is computed from the mean absolute SHAP value over a fixed holdout sample.
- Local explanations are computed for three representative holdout examples:
  - high-confidence attack
  - borderline attack
  - high-confidence normal
- A fixed random state is used so the holdout sample and outputs are reproducible.

## Outputs produced

- `output/reports/edge_iiot_shap_global_importance.csv`
- `output/reports/edge_iiot_shap_top_features.csv`
- `output/reports/edge_iiot_shap_local_examples.csv`
- `output/reports/edge_iiot_shap_local_summary.csv`
- `output/reports/edge_iiot_shap_summary.json`
- `output/figures/edge_iiot_shap_summary.png` when matplotlib is available

## Sampling and runtime tradeoffs

- SHAP is computed on a fixed sample of holdout rows rather than the full dataset to keep runtime practical.
- The default global sample size is 1000 rows.
- This is enough for a stable global ranking while avoiding the cost of explaining every holdout row.
- The local explanations are limited to a small number of hand-picked representative holdout examples.

## How this supports the paper methodology

- SHAP adds interpretability without changing the trained classifier.
- The outputs can be cited in the paper as evidence of which features drive attack predictions.
- Because the explanations are generated from the saved bundle and the same feature contract, they remain consistent with the evaluation pipeline and the baseline training path.
