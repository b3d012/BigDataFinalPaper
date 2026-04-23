# Threshold Calibration Summary

This step evaluates a fixed threshold grid on the saved holdout probabilities, and on out-of-fold CV probabilities when available.
PR-AUC is reported as context from the underlying probability scores; it does not change with threshold.

## Holdout
- Threshold grid: 0.05 to 0.95 step 0.01
- Default threshold: 0.50
- Max F1 threshold: 0.13
- Max F2 threshold: 0.13
- Best recall under precision constraint: 0.13
- Lowest FNR under precision constraint: 0.05

## CV
- Threshold grid: 0.05 to 0.95 step 0.01
- Default threshold: 0.50
- Max F1 threshold: 0.23
- Max F2 threshold: 0.23
- Best recall under precision constraint: 0.06
- Lowest FNR under precision constraint: 0.05

## Why these thresholds are reasonable
- They make the precision/recall tradeoff explicit for the binary attack detector.
- The minimum-precision constraint prevents the paper from recommending thresholds that lower false alarms too aggressively at the expense of precision.
- The lowest-FNR recommendation is useful when the paper emphasizes attack detection and missed-attack reduction.
- The max-F1 and max-F2 thresholds provide standard reference points for the tradeoff discussion.