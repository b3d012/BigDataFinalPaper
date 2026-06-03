# Step 1 Old Pipeline Audit

## What I inspected

- `BigDataFinalProject/experment/edge_iiot_experiment.py`
- `BigDataFinalProject/testoutside/live_wifi_edge_ids_pcap.py`
- `BigDataFinalProject/README.md`
- `BigDataFinalProject/EDGE_IIOT_FULL_PROJECT_GUIDE.md`
- `BigDataFinalProject/TrainingResults.md`
- `BigDataFinalProject/WORK_SUMMARY_OLDtoNEW.md`
- `BigDataFinalProject/environment-edgeids.yml`
- `src/edge_iiot_experiment.py`
- `src/feature_engineering.py`

## File inventory and role

| File | Role |
|---|---|
| `BigDataFinalProject/experment/edge_iiot_experiment.py` | Authoritative working offline classifier and offline PCAP replay utility. This is the source of truth for the paper-first rebuild. |
| `BigDataFinalProject/testoutside/live_wifi_edge_ids_pcap.py` | Live capture and live PCAP scoring utility. Useful only as a bundle/input contract reference. |
| `BigDataFinalProject/README.md` | High-level project description and run examples for train, extract, score, compare, and live monitoring. |
| `BigDataFinalProject/EDGE_IIOT_FULL_PROJECT_GUIDE.md` | Narrative explanation of the working Edge-IIoT pipeline and its outputs. |
| `BigDataFinalProject/TrainingResults.md` | Concrete runtime and evaluation numbers from the final working run. |
| `BigDataFinalProject/WORK_SUMMARY_OLDtoNEW.md` | Notes about the migration from earlier CIC attempts to the Edge-IIoT pipeline. |
| `BigDataFinalProject/environment-edgeids.yml` | Minimal environment dependencies for the working Edge-IIoT script. |
| `src/edge_iiot_experiment.py` | Legacy prototype classifier in the current repo. It uses `GradientBoostingClassifier` on `data/processed_data.csv`, not the archived working pipeline. |
| `src/feature_engineering.py` | Legacy preprocessing prototype. It applies preprocessing and SMOTE before splitting, which is not the working paper-safe baseline. |

## What the old working offline pipeline actually is

- The working pipeline is binary classification on `data/ML-EdgeIIoT-dataset.csv`.
- The model is an `XGBClassifier`, not `GradientBoostingClassifier`.
- The training script loads the Edge-IIoT CSV, normalizes column names, builds binary labels, preprocesses numeric and categorical fields, trains XGBoost, evaluates on a holdout split, then retrains the final bundle on the full dataset.
- The same saved bundle is reused for offline scoring of extracted CSVs, and the live script reuses the same bundle for window-based capture.

## Main training script

- `BigDataFinalProject/experment/edge_iiot_experiment.py`
- It provides the `train`, `extract`, `score`, `compare`, and `run` subcommands.
- For the paper-first rebuild, the `train` path is the core baseline and the `score` path is the offline evaluation companion.

## Preprocessing logic

- Column normalization removes BOM characters, trims names, and deduplicates duplicate columns.
- Duplicate rows are dropped by default.
- Raw feature columns are all non-label columns except `Attack_label`, `Attack_type`, and `__source_file__`.
- Identity and payload-heavy columns are dropped by default unless `--keep_identity_payload` is enabled.
- Numeric versus categorical typing is inferred per column using a numeric parse ratio threshold of `0.95`.
- Numeric columns use median imputation.
- Categorical columns use constant missing fill with `__MISSING__`, followed by one-hot encoding.
- Empty columns and constant columns are dropped after typing.
- The resulting `training_meta` stores the feature contract, numeric/categorical columns, parse ratios, and dropped-column lists.

## Label handling

- `Attack_label` is preferred when present.
- If `Attack_label` parses numerically, values greater than `0` become attack and the rest become normal.
- If `Attack_label` is text, values like `0`, `normal`, `benign`, and `false` are treated as normal; everything else becomes attack.
- If `Attack_label` is missing, `Attack_type` is used with the same normal-versus-attack rule.
- The target is binary, not multi-class.

## Feature handling

- The working script is packet-field oriented, not flow-CIC oriented.
- It was designed around Edge-IIoT Wireshark/tshark-style fields such as TCP, UDP, DNS, HTTP, MQTT, and ICMP packet attributes.
- The default pipeline excludes identity and payload-bearing columns to reduce memorization of the dataset environment.
- The live script reuses the saved feature contract and falls back to supported tshark fields only.

## Model type and hyperparameters

- Model type: `XGBClassifier`
- Objective: `binary:logistic`
- `n_estimators=350`
- `max_depth=6`
- `learning_rate=0.04`
- `subsample=0.85`
- `colsample_bytree=0.85`
- `reg_lambda=1.0`
- `min_child_weight=3`
- `random_state=42`
- `n_jobs=-1`
- `tree_method="hist"`
- `eval_metric="aucpr"`
- `scale_pos_weight=neg/pos` when the positive class exists

## Split and validation logic

- The script performs an 80/20 stratified train/test split.
- The train portion is then split again into train and validation with a second 80/20 stratified split.
- One model is trained on the train split to tune the record threshold on validation.
- A second model is trained on train + validation and used for the holdout test evaluation.
- A final model is trained on the full dataset and stored in the bundle.
- The validation split is used only for threshold selection, not for final reported test metrics.

## Threshold logic

- Default threshold strategy is `fixed`.
- The fixed record-level threshold defaults to `0.5`.
- Optional validation-based strategies exist for `f1` and `f2`.
- When `f1` or `f2` is selected, the threshold is chosen from the validation precision-recall curve.
- `min_precision` can filter candidate thresholds before selection.
- The live script also uses file/window-level rules built from `file_max_threshold` and `file_ratio_threshold`.
- The paper-first offline rebuild should keep the record threshold logic, but treat the live window rules as later-stage behavior.

## Saved artifacts and outputs

- Model bundle: `experment/edge_iiot_xgb_model.joblib`
- Metadata JSON: `experment/edge_iiot_xgb_model.metadata.json`
- Feature importance CSV: `experment/edge_iiot_xgb_model.feature_importance.csv`
- Offline replay predictions: `experment/edge_iiot_attack_predictions.csv`
- Offline replay summary: `experment/edge_iiot_attack_predictions_summary.csv`
- Extracted PCAP CSVs: `experment/extracted-attack-edge-csvs/`
- Live outputs: `testoutside/live-output/` or the configured output directory
- Live baseline JSON: `live_wifi_baseline.json`

## Core offline classification versus later utilities

### Core offline classification

- `train` in `BigDataFinalProject/experment/edge_iiot_experiment.py`
- `prepare_training_frame`
- `coerce_feature_types`
- `make_preprocessor`
- `train_xgb`
- `choose_threshold`
- `evaluate_predictions`
- `fit_bundle`

### Offline replay and evaluation support

- `extract` in `BigDataFinalProject/experment/edge_iiot_experiment.py`
- `score` in `BigDataFinalProject/experment/edge_iiot_experiment.py`
- `compare` in `BigDataFinalProject/experment/edge_iiot_experiment.py`
- `run` in `BigDataFinalProject/experment/edge_iiot_experiment.py`

### Later live/deployment utilities

- `BigDataFinalProject/testoutside/live_wifi_edge_ids_pcap.py`
- Live capture from an interface
- PCAP-folder monitor mode
- Baseline calibration JSON generation
- Window-level alert logic
- Packet-level live output CSVs

## What is reusable right now

- The archived Edge-IIoT training flow.
- The label-building logic.
- The feature typing and preprocessing rules.
- The XGBoost hyperparameters.
- The holdout evaluation pattern and saved bundle format.
- The offline CSV scoring path for model verification.

## What should be ignored for now

- Live capture and interface monitoring.
- Tshark install discovery and baseline calibration.
- PCAP capture loops and deployment wrappers.
- MongoDB, Spark, streaming, and dashboard ideas.
- The legacy `src/feature_engineering.py` SMOTE-before-split prototype.
- The root `src/edge_iiot_experiment.py` GradientBoosting prototype.

## Risks and mismatches versus the paper plan

- The archived pipeline is binary attack-versus-normal classification, while the paper may later want per-class reporting beyond the binary label.
- The archived evaluation prints per-class precision and recall in `classification_report`, but FNR is not explicitly emitted and must be derived from the confusion matrix.
- The archived live utilities introduce an additional window/file threshold layer that is not part of the first offline-only rebuild.
- The old working pipeline is dataset-specific to Edge-IIoT packet fields, so a later paper version must be careful when comparing it against CIC-style flow features.
- The SMOTE prototype in `src/feature_engineering.py` is not a safe baseline because it resamples before the split.

## Bottom line

- Use `BigDataFinalProject/experment/edge_iiot_experiment.py` as the reference implementation for the first offline paper rebuild.
- Treat `src/edge_iiot_experiment.py` and `src/feature_engineering.py` as historical prototypes only.
