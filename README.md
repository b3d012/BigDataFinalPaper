# Explainable Edge-IIoT Intrusion Detection

This repository contains the implementation for an explainable Edge-IIoT intrusion detection system. The system trains an XGBoost binary IDS, evaluates training-only SMOTE cross-validation, calibrates decision thresholds, trains an attack-subtype classifier, generates SHAP explanations, adds benign-anomaly and drift monitoring layers, and exposes the evidence through a MongoDB-backed Streamlit dashboard.

The final paper PDF and Word document are submitted separately. This GitHub repository is intentionally kept code-focused and lightweight: large datasets, PCAPs, model binaries, dashboard screenshots, live-capture logs, and LaTeX build outputs are not committed.

## Highlights

- Binary XGBoost detector for Edge-IIoT normal/attack detection.
- Training-only SMOTE cross-validation to reduce missed attacks without validation leakage.
- Attack-subtype XGBoost model trained from Edge-IIoT `Attack_type` labels.
- Threshold calibration using PR-AUC, ROC-AUC, precision, recall, FNR, and confusion counts.
- SHAP global importance and top-3 local alert explanations.
- Isolation Forest benign-anomaly override for suspicious benign predictions.
- ADWIN live drift monitoring with validation-gated retraining as a backup mechanism.
- MongoDB and Streamlit dashboard for offline artifacts, live windows, PCAP replay, SHAP, drift, retraining, and latency evidence.

## Repository Layout

- `src/` - runnable implementation scripts.
- `tests/` - unit tests for runtime contracts, live hardening, Mongo schema, ADWIN, multiclass thresholds, and robustness helpers.
- `docs/` - implementation notes and verification notes.
- `schemas/` - MongoDB prediction/evidence schema.
- `models/*.json` - lightweight model metadata and feature contract files.
- `output/reports/*.json` and `output/reports/*.md` - lightweight result summaries used by the paper and dashboard.
- `output/demo/*.json` and `output/demo/*.md` - lightweight demo replay summaries.
- `requirements.txt` and `environment.yml` - reproducible Python environment files.

Large local files are expected but ignored by Git:

- `data/ML-EdgeIIoT-dataset.csv`
- `demo/*.pcap`, `demo/*.pcapng`, or `demo/*.cap`
- `models/*.joblib`
- `output/figures/`
- `output/live/`
- generated CSV prediction dumps and LaTeX/PDF/Word build outputs

## Setup

Use the Conda environment when possible:

```powershell
conda env create -f environment.yml
conda activate bigdatafinalpaper
```

Or use pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional external tools:

- Wireshark/TShark for PCAP replay and live capture.
- MongoDB for dashboard persistence and live evidence.

## Required Local Inputs

Place the Edge-IIoT CSV here:

```text
data/ML-EdgeIIoT-dataset.csv
```

Place optional demo PCAPs here:

```text
demo/
```

The scripts recreate model binaries and generated outputs locally. The GitHub repository stores only metadata and compact report summaries.

## Reproduction Workflow

Run from the repository root.

```powershell
python src\edge_iiot_experiment.py train
python src\edge_iiot_experiment.py train --use_smote --cv_folds 5
python src\edge_iiot_experiment.py train_multiclass
python src\edge_iiot_shap.py
python src\edge_iiot_thresholds.py
python src\edge_iiot_anomaly.py train
python src\edge_iiot_drift.py demo
python src\edge_iiot_retrain.py train
python src\edge_iiot_robustness.py
```

Seed MongoDB and start the dashboard:

```powershell
python src\edge_iiot_mongo.py seed
streamlit run src\edge_iiot_dashboard.py
```

Run live capture directly, or use the dashboard controls:

```powershell
python src\edge_iiot_live_capture.py live --interface 5 --window_seconds 30
```

## Validation

```powershell
python -m pytest -q
python -m compileall -q src tests
```

The tests focus on feature-contract validation, leakage-safe SMOTE behavior, live-path hardening, top-3 SHAP alert payloads, MongoDB traceability schema, ADWIN events, multiclass thresholds, and robustness report generation.

## Result Summary

The committed summaries document the following verified results:

- Holdout binary detector: 99.26% ROC-AUC, 99.85% PR-AUC, 94.73% attack recall at threshold 0.5.
- SMOTE five-fold cross-validation: 99.98% mean attack recall and 0.025% mean attack FNR.
- Attack-subtype model: 14 Edge-IIoT attack classes, 88.52% macro recall, 89.82% weighted F1.
- SHAP explainability: global importance plus exactly top-3 local contributors for alert records.
- Live/demo verification: MongoDB prediction traceability, alert latency fields, anomaly overrides, ADWIN drift events, retraining request/result records, and dashboard views.

## Notes

The original binary model is treated as the primary detector. Adaptive retraining is implemented as a guarded backup path: drift can trigger a candidate model, but the active pointer should change only if validation metrics justify replacement.
