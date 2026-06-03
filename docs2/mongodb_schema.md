# MongoDB Schema

The dashboard and live worker write the following collections to `edge_iiot_paper`.

| Collection | Purpose | Key fields |
|---|---|---|
| `raw_packets` | Raw packet captures from live windows and PCAP injection | `kind`, `source_file`, `record_index` |
| `feature_vectors` | Transformed feature rows used for scoring and retraining | `kind`, `source_file`, `record_index` |
| `predictions` | Binary predictions, anomaly predictions, and multiclass outputs | `kind`, `source`, `source_file`, `record_index` |
| `alert_explanations` | Per-alert SHAP explanations for each scored row | `kind`, `source`, `source_file`, `window_id`, `record_index` |
| `live_windows` | Window-level summaries with SHAP and drift metadata | `kind`, `source_file`, `window_id` |
| `feature_importance` | Global and live feature importance summaries | `importance_type`, `source`, `feature` |
| `threshold_metrics` | Threshold grids and calibration metrics | `kind`, `source`, `threshold` |
| `drift_events` | ADWIN drift detections on prediction confidence streams | `kind`, `source`, `window_id`, `record_index`, `created_at` |
| `retrain_events` | Retraining requests and retraining outcomes | `kind`, `source`, `created_at` |
| `adversarial_evaluations` | FGSM / PGD robustness results | `kind`, `source`, `created_at` |
| `analysis_summaries` | Saved summaries, run reports, and feature contracts | `kind`, `artifact_path` |

## Document shape notes

- Prediction documents store `pred_proba_attack`, `pred_label`, `model_version`, `attack_model_version`, and optional `attack_type_name`.
- Alert explanations store `prediction_id`, `top_features`, `base_value`, `expected_value`, and `raw_probability`.
- Drift events store the ADWIN state snapshot plus the confidence value that triggered the update.
- Retrain events store request/result records so the dashboard can show trigger reason, exit code, and acceptance.
