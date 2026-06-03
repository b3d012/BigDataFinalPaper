# Step 6 Demo PCAP Replay

## What was added

- Added an offline demo replay script: `src/edge_iiot_demo_replay.py`.
- The script replays PCAPs from `demo/`, extracts tshark fields, scores the rows with the saved Edge-IIoT bundle, and aggregates file-level predictions.
- It also compares the new file summary against the old reference summary CSV placed in `demo/`.

## How demo PCAPs are processed

- The script scans `demo/` for `.pcap`, `.pcapng`, and `.cap` files.
- It requests the model feature contract from `models/edge_iiot_xgb_model.joblib`.
- tshark extracts those fields into `output/demo/extracted_csvs/`.
- The extracted rows are transformed using the saved preprocessing contract and scored with the saved threshold.
- File-level summaries are computed from the per-record probabilities and labels.

## Files produced

- `output/demo/extracted_csvs/`
- `output/demo/edge_iiot_demo_predictions.csv`
- `output/demo/edge_iiot_demo_predictions_summary.csv`
- `output/demo/edge_iiot_demo_vs_reference_comparison.csv`
- `output/demo/edge_iiot_demo_run_summary.md`

## How comparison works

- The new summary is matched against the old reference summary by filename.
- The comparison report shows:
  - whether a file was present in both summaries
  - whether `file_pred_label` matched
  - deltas for the probability and ratio columns
  - which files matched exactly and which did not
- If the reference summary is missing, the replay still runs and the comparison step is skipped with a clear message.

## Limitations

- This is a replay and comparison workflow, not live capture.
- Parity with the old project depends on tshark field availability and the fact that the current model bundle is different from the legacy model.
- Exact file-label parity is not guaranteed if the old and new thresholds or model versions differ.
- The script remains offline-only and intentionally avoids streaming, dashboards, and any capture loop behavior.
