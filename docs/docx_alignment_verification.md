# DOCX Alignment Verification

Date: 2026-05-03

The original methodology document referenced an older IoMT dataset. This project now uses Edge-IIoT as the replacement dataset, so class names and counts are mapped to Edge-IIoT rather than copied from the older dataset.

## Implemented Methodology Mapping

| DOCX requirement | Project evidence |
| --- | --- |
| Binary attack detector | `models/edge_iiot_xgb_model.joblib` and active pointer in `models/edge_iiot_active_model.json` |
| Attack-subtype detector | `models/edge_iiot_attack_xgb_model.joblib`, 14 Edge-IIoT attack classes |
| Feature contract | `models/edge_iiot_feature_contract.json`, validated before inference |
| Training-only SMOTE | Five-fold CV applies SMOTE inside each training fold only |
| PR-AUC and recall-focused metrics | Holdout, CV, threshold, retrain, and multiclass reports prioritize PR-AUC, recall, precision, FNR, and confusion counts |
| SHAP explainability | Global SHAP artifacts plus live top-3 SHAP alert payloads |
| Isolation Forest wrapper | Benign-only model flags benign predictions as `Potential Evasion/Anomalous` |
| Drift detection | PSI offline reports and ADWIN live events with `delta=0.002` |
| Adaptive retraining | Live ADWIN requests retraining; candidates must pass PR-AUC, attack recall, and normal recall gates |
| MongoDB traceability | Prediction records include flow data, prediction, SHAP payload, drift status, anomaly status, model version, and latency fields |
| Dashboard/demo stack | Existing `tshark + MongoDB + Streamlit` stack, not a new Spark/Kafka stack |
| Adversarial robustness | `output/reports/edge_iiot_adversarial_robustness.json` contains FGSM and PGD outputs |

## Verification Commands

```powershell
python -m pytest -q
python -m compileall -q src tests
tshark -D
python src\edge_iiot_experiment.py train_multiclass
python src\edge_iiot_thresholds.py --multiclass_predictions output\reports\edge_iiot_multiclass_holdout_predictions.csv --multiclass_output output\reports\edge_iiot_multiclass_thresholds.json
python src\edge_iiot_mongo.py seed
python src\edge_iiot_live_capture.py live --interface 5 --tshark "C:\Program Files\Wireshark\tshark.exe" --max_windows 1 --window_seconds 1 --packet_count 1 --validate_tshark_fields --mongo_uri mongodb://localhost:27017 --db_name edge_iiot_paper
python src\edge_iiot_live_capture.py live --interface 5 --tshark "C:\Program Files\Wireshark\tshark.exe" --inbox_dir output\live\inbox --max_windows 5 --window_seconds 1 --packet_count 200 --mongo_uri mongodb://localhost:27017 --db_name edge_iiot_paper
python src\edge_iiot_mongo.py status
```

## Key Results

| Area | Result |
| --- | --- |
| Multiclass attack model | 14 classes, macro recall 88.52%, weighted F1 89.82% |
| Critical attack recalls | DDoS TCP 100%, DDoS UDP 100%, DDoS ICMP 99.89%, DDoS HTTP 90.34%, MITM 100%, Ransomware 88.28% |
| Robustness | Clean attack recall 93.60%; FGSM and PGD attack recall 92.80% |
| MongoDB counts after verification | 172,781 raw packets, 172,781 feature vectors, 564,033 predictions, 316 alert explanations, 17 live windows, 10 drift events, 18 retrain events |
| Live accepted retrain | One ADWIN-triggered candidate passed the gate and updated the active pointer |
| Live rejected retrains | Later candidates were rejected for `normal_recall<0.50` |

## Live PCAP Windows

| PCAP window | Rows | Attack rows | Anomaly overrides | Window label |
| --- | ---: | ---: | ---: | --- |
| BENIGN_BASELINE | 200 | 54 | 100 | Normal |
| NMAP_SYN_SCAN | 200 | 126 | 50 | Attack |
| SSH_BRUTEFORCE_PATTERN | 200 | 190 | 3 | Attack |
| HTTP_FLOOD_PATTERN | 200 | 194 | 1 | Attack |
| UDP_FLOOD_PATTERN | 200 | 163 | 24 | Attack |

## Submission Caveats

The demo stack is local and reproducible, but it is not a distributed Spark/Kafka deployment. Replayed PCAPs expose packet fields through `tshark`; fields absent from live captures are imputed by the saved preprocessing contract, which is expected for this dataset-to-PCAP demonstration.
