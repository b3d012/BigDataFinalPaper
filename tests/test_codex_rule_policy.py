from __future__ import annotations

from pathlib import Path


def test_active_binary_evaluation_does_not_use_accuracy_score() -> None:
    for path in [
        Path("src/edge_iiot_experiment.py"),
        Path("src/edge_iiot_thresholds.py"),
        Path("src/edge_iiot_retrain.py"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "accuracy_score" not in text
        assert "\"accuracy\"" not in text
        assert "'accuracy'" not in text


def test_smote_is_training_fold_only_by_construction() -> None:
    text = Path("src/edge_iiot_experiment.py").read_text(encoding="utf-8")
    assert "smote.fit_resample(X_train_tx, y_series)" in text
    assert "smote.fit_resample(X_val_tx" not in text
    assert '"smote_training_only": True' in text


def test_live_defaults_enable_retrain_and_top3_shap() -> None:
    text = Path("src/edge_iiot_live_capture.py").read_text(encoding="utf-8")
    assert 'live_parser.add_argument("--shap_top_n", type=int, default=3)' in text
    assert 'live_parser.add_argument("--auto_retrain", action=argparse.BooleanOptionalAction, default=True)' in text
    assert ".head(3)" in text
