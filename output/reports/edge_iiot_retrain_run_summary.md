# Edge-IIoT Adaptive Retraining Summary

- Retraining triggered: True
- Trigger reason: severity high (max PSI 1.5963 >= 0.25)
- Trigger severity: high
- Reference rows: 94192
- Drift rows: 31397
- Evaluation rows: 31397
- Split strategy: contiguous_60_20_20

## Before / After Metrics
- accuracy: original=0.994012, retrained=0.992165, delta=-0.001847
- roc_auc: original=0.999993, retrained=0.993235, delta=-0.006757
- pr_auc: original=0.999999, retrained=0.998776, delta=-0.001224
- attack_precision: original=0.993467, retrained=0.991468, delta=-0.001998
- attack_recall: original=1.000000, retrained=1.000000, delta=0.000000
- attack_fnr: original=0.000000, retrained=0.000000, delta=0.000000

## Assessment
- Overall retraining assessment: hurt
- Retrained bundle: models\edge_iiot_xgb_model_retrained.joblib