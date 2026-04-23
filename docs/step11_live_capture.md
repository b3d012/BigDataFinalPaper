# Step 11 Live Capture

This step adds a live capture mode to the Streamlit dashboard. It keeps the offline pipeline unchanged and uses a separate tshark-backed worker process to capture packets, score them, and write results into MongoDB.

## What Was Added

- Start Capture and Stop Capture controls in the dashboard
- Configurable live capture windows
- A background worker: `src/edge_iiot_live_capture.py`
- MongoDB storage for live packets, feature vectors, predictions, window summaries, and live SHAP-style feature contributions
- Auto-refreshing dashboard panels for live status and scores

## How Live Capture Works

The dashboard launches a separate Python worker process when you click Start Capture. The worker:

- finds `tshark`
- captures packets from the selected interface
- extracts the same feature contract used by the trained XGBoost model
- scores each packet with `models/edge_iiot_xgb_model.joblib`
- writes raw rows, features, predictions, and window summaries to MongoDB
- optionally computes live SHAP-style feature contributions from the XGBoost booster

The worker command used by the dashboard is conceptually:

```powershell
python src\edge_iiot_live_capture.py live --interface <iface> --tshark <path-or-tshark> --window_seconds 30
```

The dashboard passes the selected filters and thresholds to this worker automatically.

## Window Size Controls

The dashboard exposes these live capture controls:

- `Capture window seconds`
- `Packet count per window`
- `Capture filter`
- `Display filter`
- `Pause between windows`
- `Max windows`

This lets you choose either time-based windows or packet-count-limited windows, depending on the demo you want to show.

## Real-Time Predictions

The live panel updates from MongoDB and shows:

- records captured in the latest window
- attack ratio in the latest window
- mean and max attack probability
- the window-level attack decision
- a probability trace for the latest window
- a packet table with live prediction outputs

The live worker also stores per-packet predictions in MongoDB, so the dashboard can render recent live rows directly.

## Real-Time SHAP And Threshold Analysis

For each live window, the worker stores live feature-contribution summaries. The dashboard uses those records to show:

- a live SHAP-style feature-importance bar chart
- the top features for the most recent window
- a threshold slider that dynamically recomputes the live attack flag from stored probabilities

## Metrics Shown

The dashboard always shows score-based live metrics:

- records
- attack ratio
- mean attack probability
- max attack probability
- window decision label

If labeled rows are available, the dashboard also computes:

- accuracy
- precision
- recall
- FNR
- ROC-AUC

For normal live capture, labels are usually not available. In that case, the metric cards remain visible, but the labeled metrics are shown as `n/a`.

## Offline Fallback

If live capture is not started, or `tshark` is unavailable, the dashboard still runs using the saved offline MongoDB collections and report files.

## Limitations

- This is a dashboard-driven live demo, not a production streaming system.
- Live accuracy/precision/recall/FNR require labels; true live traffic usually does not provide them.
- The live capture worker depends on `tshark` and the local capture permissions for the selected interface.
- The current implementation updates in capture windows rather than per-packet micro-batches.

## Paper Relevance

This step shows how the offline Edge-IIoT pipeline can be extended to a live operational view without changing the classifier itself. It also demonstrates how the same bundle can support capture, scoring, SHAP-style explanations, threshold tuning, and MongoDB-backed visualization.
