# Edge-IIoT Adaptive Retraining Summary

- Retraining triggered: True
- Trigger reason: severity high (max PSI 1.5963 >= 0.25)
- Trigger severity: high
- Reference rows: 94192
- Drift rows: 31397
- Evaluation rows: 31397
- Split strategy: contiguous_60_20_20

## Before / After Metrics
- roc_auc: original=0.993235, retrained=0.993235, delta=0.000000
- pr_auc: original=0.998776, retrained=0.998776, delta=0.000000
- attack_precision: original=0.991468, retrained=0.991468, delta=0.000000
- attack_recall: original=1.000000, retrained=1.000000, delta=0.000000
- normal_recall: original=0.912424, retrained=0.912424, delta=0.000000
- macro_recall: original=0.956212, retrained=0.956212, delta=0.000000
- attack_fnr: original=0.000000, retrained=0.000000, delta=0.000000

## Assessment
- Overall retraining assessment: little_difference
- Retrained bundle: C:\Users\abdul\Desktop\BigDataFinalPaper\models\edge_iiot_xgb_model_retrained.joblib