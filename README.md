# BigDataFinalPaper

Offline Edge-IIoT IDS rebuild for the paper-oriented workflow.

This repository is set up so someone else can clone it, install the Python environment, place the dataset and demo files in the expected folders, and run the existing scripts as-is.

## Repository Layout

- `data/` - main dataset location. The primary file used by the scripts is `data/ML-EdgeIIoT-dataset.csv`.
- `demo/` - demo PCAP files used for replay, comparison, and injected live windows.
- `docs/` - step-by-step notes for each stage of the rebuild.
- `models/` - saved model bundles and metadata files.
- `notebooks/` - optional workspace for exploratory notebooks. This folder is a scratch area for analysis; no notebook is required to run the pipeline.
- `output/` - generated reports, figures, live capture output, and dashboard artifacts.
- `src/` - all runnable scripts.
- `BigDataFinalProject/` - archived reference project kept for comparison only.

## Setup

### 1. Install Python

Use Python 3.11 if possible. The original reference environment was based on 3.11.

### 2. Create an environment

You can use either Conda or `venv`.

#### Option A: Conda from the root environment file

```powershell
conda env create -f environment.yml
conda activate bigdatafinalpaper
```

#### Option B: Python `venv` from the root requirements file

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Option C: Archived reference environment

The archived project still includes the original starter environment file at `BigDataFinalProject\environment-edgeids.yml`.

```powershell
conda env create -f BigDataFinalProject\environment-edgeids.yml
conda activate edgeids
```

Then install the extra packages used by the current rebuild:

```powershell
python -m pip install shap matplotlib imbalanced-learn streamlit plotly streamlit-autorefresh pymongo
```

### 3. Install external tools for live/demo work

These are only needed for the optional live and replay workflows:

- Install Wireshark so `tshark` is available.
- Install MongoDB locally, or point the dashboard to MongoDB Atlas with `MONGODB_URI`.

If `tshark` is not on your `PATH`, the live and demo replay scripts let you pass the full executable path.

### 4. Confirm the required files are present

Before running the scripts, make sure these files/folders exist:

- `data\ML-EdgeIIoT-dataset.csv`
- `demo\` with any demo `.pcap`, `.pcapng`, or `.cap` files you want to replay
- `models\` and `output\` will be created or updated by the scripts as needed

## Recommended Workflow

If you are reproducing the project from scratch, run the scripts in this order:

1. Offline baseline classifier
2. SMOTE + cross-validation
3. SHAP explainability
4. Threshold calibration
5. Anomaly detector
6. Drift detection
7. Adaptive retraining
8. Demo PCAP replay
9. MongoDB seeding and dashboard
10. Live capture / live dashboard

The sections below show the exact commands.

## Offline Classifier

### Baseline training

```powershell
python src\edge_iiot_experiment.py train
```

This is the main binary Edge-IIoT holdout baseline. It uses the archived preprocessing contract, the archived XGBoost settings, a fixed threshold, and holdout evaluation.

### SMOTE + 5-fold CV

```powershell
python src\edge_iiot_experiment.py train --use_smote --cv_folds 5
```

This keeps the baseline path and adds training-only SMOTE inside each fold, then evaluates a stratified 5-fold cross-validation run.

## Explainability and Calibration

### SHAP explainability

```powershell
python src\edge_iiot_shap.py
```

Generates global SHAP importance, top features, local examples, and a summary figure from the saved model bundle.

### Threshold calibration

```powershell
python src\edge_iiot_thresholds.py
```

Builds threshold grids and recommendation files for the holdout and CV outputs. The live dashboard can also surface live threshold rows when they exist in MongoDB.

## Robustness Layers

### Anomaly detection

```powershell
python src\edge_iiot_anomaly.py train
```

Trains the Isolation Forest anomaly layer on benign training rows only and evaluates it on the held-out split.

### Drift detection

```powershell
python src\edge_iiot_drift.py dataset
python src\edge_iiot_drift.py demo
```

Compares two batches in the transformed feature space using PSI-style drift scoring.

### Adaptive retraining

```powershell
python src\edge_iiot_retrain.py train
```

Uses the drift results to decide whether retraining should be triggered, then compares the original model and retrained model on the same evaluation target.

## Demo Replay

### Offline PCAP replay

```powershell
python src\edge_iiot_demo_replay.py
```

Replays `.pcap`, `.pcapng`, or `.cap` files from `demo/`, extracts tshark fields, scores them with the saved bundle, and compares the per-file summary against the archived reference output.

## MongoDB and Dashboard

### Seed MongoDB from saved artifacts

```powershell
python src\edge_iiot_mongo.py seed
```

This loads the offline outputs into MongoDB so the dashboard can read them. Use `MONGODB_URI` and `MONGODB_DB` if you want to point at a different database.

### Start the dashboard

```powershell
streamlit run src\edge_iiot_dashboard.py
```

The dashboard reads MongoDB first and falls back to the saved CSV/JSON artifacts when MongoDB is not available.

### Live capture

The dashboard can start and stop the live capture worker, which uses `tshark` and writes live rows into MongoDB.

Live capture is controlled from the dashboard, but the worker can also be run directly:

```powershell
python src\edge_iiot_live_capture.py live --interface <iface> --tshark <path-or-tshark> --window_seconds 30
```

Live capture writes into `output\live\` and stores live packet rows, predictions, SHAP rows, drift summaries, and live window summaries in MongoDB.

## What the Scripts Expect

- The classifier scripts expect `data\ML-EdgeIIoT-dataset.csv`.
- The replay and live scripts expect `demo\` files or a live network interface.
- The dashboard expects MongoDB if you want the live views, but it still works with the saved offline artifacts.
- The notebooks folder is available for exploratory work, but the main pipeline does not depend on any notebook.

## Generated Outputs

The main outputs are written under `models\`, `output\reports\`, `output\figures\`, `output\demo\`, and `output\live\`.

Common examples:

- `models\edge_iiot_xgb_model.joblib`
- `models\edge_iiot_xgb_model.metadata.json`
- `models\edge_iiot_xgb_model.feature_importance.csv`
- `output\reports\edge_iiot_holdout_metrics.json`
- `output\reports\edge_iiot_cv_summary.json`
- `output\reports\edge_iiot_shap_summary.json`
- `output\reports\edge_iiot_holdout_threshold_recommendations.json`
- `output\reports\edge_iiot_anomaly_holdout_metrics.json`
- `output\reports\edge_iiot_drift_summary.json`
- `output\reports\edge_iiot_retrain_comparison.json`
- `output\demo\edge_iiot_demo_predictions.csv`
- `output\demo\edge_iiot_demo_anomaly_summary.csv`
- `output\demo\edge_iiot_demo_drift_summary.json`
- `output\live\live_capture_status.json`

## Notes for Contributors

- Keep new analysis notebooks in `notebooks\` if you want to do ad hoc exploration.
- Do not overwrite the archived `BigDataFinalProject\` folder; it is only there for reference.
- If you add new dependencies, update this README with the install command so the setup stays reproducible.
