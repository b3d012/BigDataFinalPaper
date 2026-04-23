# Step 1 Recommended Structure

## Goal

- Rebuild the first paper-ready version as a minimal offline Edge-IIoT classifier.
- Keep the implementation close to the archived working pipeline.
- Defer live capture, PCAP replay automation, anomaly detection, SHAP, drift handling, and deployment until the offline baseline is stable.

## Minimal repo structure

```text
BigDataFinalPaper/
├── data/
│   ├── ML-EdgeIIoT-dataset.csv
│   ├── raw/
│   └── processed/
├── docs/
│   ├── step1_old_pipeline_audit.md
│   └── step1_recommended_structure.md
├── models/
├── notebooks/
├── output/
│   ├── predictions/
│   ├── reports/
│   └── feature_importance/
└── src/
    └── edge_iiot_experiment.py
```

## Current folders that can be reused

- `data/` for the Edge-IIoT CSV and any later processed extracts.
- `models/` for the saved joblib bundle, metadata JSON, and feature importance CSV.
- `output/` for training reports, predictions, confusion matrices, and paper figures.
- `docs/` for the audit, implementation notes, and paper-method notes.
- `notebooks/` for exploratory analysis only, not as the source of truth.
- `src/` for the canonical offline training and evaluation script.

## New scripts that should exist first

- `src/edge_iiot_experiment.py`; this should be the canonical offline trainer/evaluator rebuilt from the archived XGBoost pipeline.
- `src/edge_iiot_data.py`; only add this if data loading and feature typing start to become too large for one file.
- `src/edge_iiot_metrics.py`; only add this if evaluation and reporting need to be shared across scripts.
- `src/edge_iiot_bundle_io.py`; only add this if bundle save/load logic starts to repeat.

## Recommended order of implementation

1. Rebuild `src/edge_iiot_experiment.py` as the offline dataset-only pipeline.
2. Make training output a reproducible saved bundle and metadata file.
3. Add offline scoring against held-out or extracted CSVs only after the training path is stable.
4. Add richer evaluation outputs such as per-class recall, FNR, and confusion matrices in saved reports.
5. Add SHAP explainability after the baseline metrics are validated.
6. Add threshold calibration, drift detection, and anomaly detection as separate follow-on modules.
7. Add live capture, PCAP monitoring, and deployment artifacts only if the paper later needs them.

## What the first rebuild should contain

- Dataset-only loading from `data/ML-EdgeIIoT-dataset.csv`.
- Binary label construction from `Attack_label` or `Attack_type`.
- The Edge-IIoT preprocessing contract from the archived script.
- Stratified train/validation/test splitting.
- XGBoost training with the archived hyperparameters.
- PR-AUC, confusion matrix, per-class recall, and FNR reporting.
- A saved bundle containing the model, preprocessor, threshold, and training metadata.

## What should stay out of the first rebuild

- Live traffic capture.
- Tshark interface management.
- PCAP extraction loops.
- Baseline calibration JSON.
- Spark, MongoDB, and dashboard code.
- SMOTE-before-split preprocessing.
- CICFlowMeter-specific flow logic.

## Assumptions

- `BigDataFinalProject/experment/edge_iiot_experiment.py` is the correct historical baseline to port first.
- `src/edge_iiot_experiment.py` should be treated as the active rebuild target in the current repository.
- `src/feature_engineering.py` is a prototype only and should not define the paper baseline.
- The paper can grow from a stable binary offline classifier before adding calibration, drift handling, and live deployment.
