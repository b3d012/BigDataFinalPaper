# Step 10 MongoDB and Dashboard

This step adds a MongoDB-backed storage layer and a Streamlit dashboard for the paper pipeline. It is still aligned with the offline-first project structure, but it creates the bridge needed for storing results and visualizing them in one place.

## What Was Added

- `src/edge_iiot_mongo.py`
- `src/edge_iiot_dashboard.py`
- MongoDB seed support for the current offline artifacts
- A dashboard that reads MongoDB first and falls back to the saved report files when MongoDB is unavailable

## MongoDB Setup

The default connection uses:

- URI: `mongodb://localhost:27017`
- Database: `edge_iiot_paper`

You can override both with environment variables:

- `MONGODB_URI`
- `MONGODB_DB`

Current repo state assumes a local MongoDB service is available. If you want Atlas instead, set `MONGODB_URI` to the Atlas connection string and keep the rest of the code unchanged.

Seed the current offline outputs into MongoDB with:

```powershell
python src\edge_iiot_mongo.py seed
```

## Collection Schema

The code uses these collections:

- `raw_packets`
- `feature_vectors`
- `predictions`
- `feature_importance`
- `threshold_metrics`
- `analysis_summaries`

Typical document shapes are:

- `raw_packets`
  - `kind`
  - `source`
  - `source_file`
  - `record_index`
  - `created_at`
  - raw packet or extracted-row payload
- `feature_vectors`
  - same metadata as `raw_packets`
  - extracted feature payload
- `predictions`
  - `kind`
  - `source`
  - `source_file`
  - `record_index`
  - `true_label`
  - `pred_proba_attack`
  - `pred_label`
  - `threshold`
  - optional `fold`
- `feature_importance`
  - `importance_type`
  - `source`
  - `feature`
  - numeric importance values such as SHAP or drift scores
- `threshold_metrics`
  - `kind`
  - `source`
  - `threshold`
  - precision/recall/FNR and related values
- `analysis_summaries`
  - `kind`
  - `artifact_path`
  - `payload`

## Dashboard Setup

Run the Streamlit dashboard with:

```powershell
streamlit run src\edge_iiot_dashboard.py
```

The dashboard provides:

- model performance summary cards
- prediction tables and probability trends
- SHAP and other feature-importance plots
- threshold calibration plots and recommendation tables
- anomaly, drift, and retraining summaries

## What The Dashboard Displays

- Performance over time or sequence for classifier and anomaly predictions
- Latest incoming prediction rows from MongoDB
- SHAP global importance and top feature tables
- Threshold precision/recall/FNR tradeoff curves
- Anomaly recall/FNR and score summaries
- Drift severity and top drifted features
- Retraining trigger reason and before/after metric comparison

## Limitations

- This step does not add live packet capture or a continuously running ingestion service.
- The repo seeds MongoDB from the existing offline outputs so the dashboard is usable immediately.
- Raw packet storage is represented by the demo extracted rows in this repository; live packet documents can be added later with the same schema.
- If MongoDB is not running, the dashboard falls back to the saved report files instead of failing hard.

## Paper Relevance

This step supports the paper by showing how model outputs, thresholds, explainability results, anomaly scores, drift results, and retraining comparisons can be persisted and inspected from a single interactive interface.
