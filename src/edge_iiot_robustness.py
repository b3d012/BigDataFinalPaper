from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split

from edge_iiot_anomaly import load_bundle as load_anomaly_bundle
from edge_iiot_experiment import DEFAULT_EDGE_CSV, prepare_training_frame
from edge_iiot_runtime import json_safe, write_json


def load_classifier_bundle(model_path: str | Path) -> dict[str, Any]:
    import joblib

    bundle = joblib.load(model_path)
    required = {"model", "preprocessor", "training_meta"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Classifier bundle is missing required keys: {sorted(missing)}")
    return bundle


def _fit_surrogate(X: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> SGDClassifier:
    surrogate = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-4,
        random_state=random_state,
    )
    surrogate.fit(X, y)
    return surrogate


def _dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _clip_to_bounds(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(values, lower, upper)


def _fgsm_attack(
    X: np.ndarray,
    surrogate: SGDClassifier,
    *,
    epsilon: float,
    target_positive: bool = False,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    coef = surrogate.coef_[0]
    direction = np.sign(coef)
    if target_positive:
        direction = -direction
    adv = X + epsilon * direction
    if lower is not None and upper is not None:
        adv = _clip_to_bounds(adv, lower, upper)
    return adv


def _pgd_attack(
    X: np.ndarray,
    surrogate: SGDClassifier,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    target_positive: bool = False,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    coef = surrogate.coef_[0]
    direction = np.sign(coef)
    if target_positive:
        direction = -direction
    original = X.copy()
    adv = X.copy()
    for _ in range(max(1, steps)):
        adv = adv + step_size * direction
        perturb = np.clip(adv - original, -epsilon, epsilon)
        adv = original + perturb
        if lower is not None and upper is not None:
            adv = _clip_to_bounds(adv, lower, upper)
    return adv


def _classify_with_bundle(bundle: dict[str, Any], X_raw: pd.DataFrame) -> np.ndarray:
    tx = bundle["preprocessor"].transform(X_raw)
    return bundle["model"].predict_proba(tx)[:, 1]


def evaluate_robustness(
    *,
    classifier_bundle_path: str | Path,
    anomaly_bundle_path: str | Path | None = None,
    edge_csv: str | Path = DEFAULT_EDGE_CSV,
    sample_rows: int | None = None,
    epsilon: float = 0.05,
    pgd_steps: int = 5,
    pgd_step_size: float = 0.01,
    output_path: str | Path = "output/reports/edge_iiot_adversarial_robustness.json",
) -> dict[str, Any]:
    classifier_bundle = load_classifier_bundle(classifier_bundle_path)
    anomaly_bundle = load_anomaly_bundle(Path(anomaly_bundle_path)) if anomaly_bundle_path else None

    X, y, training_meta = prepare_training_frame(
        edge_csv,
        keep_identity_payload=False,
        numeric_threshold=0.95,
        sample_rows=sample_rows,
        drop_duplicates=True,
    )
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    surrogate_X = classifier_bundle["preprocessor"].transform(X_train)
    surrogate = _fit_surrogate(_dense(surrogate_X), y_train.to_numpy())

    test_tx = classifier_bundle["preprocessor"].transform(X_test)
    test_dense = _dense(test_tx)
    lower = np.nanmin(test_dense, axis=0)
    upper = np.nanmax(test_dense, axis=0)

    clean_proba = classifier_bundle["model"].predict_proba(test_tx)[:, 1]
    clean_pred = (clean_proba >= float(classifier_bundle.get("threshold", 0.5))).astype(int)
    clean_cm = confusion_matrix(y_test, clean_pred, labels=[0, 1])

    attack_mask = y_test.to_numpy() == 1
    attack_dense = test_dense[attack_mask]
    attack_true = y_test.to_numpy()[attack_mask]
    if attack_dense.size == 0:
        raise ValueError("No attack samples available for robustness evaluation.")

    fgsm_adv = _fgsm_attack(attack_dense, surrogate, epsilon=epsilon, target_positive=False, lower=lower, upper=upper)
    pgd_adv = _pgd_attack(
        attack_dense,
        surrogate,
        epsilon=epsilon,
        step_size=pgd_step_size,
        steps=pgd_steps,
        target_positive=False,
        lower=lower,
        upper=upper,
    )

    def score_adv(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proba = classifier_bundle["model"].predict_proba(matrix)[:, 1]
        pred = (proba >= float(classifier_bundle.get("threshold", 0.5))).astype(int)
        return proba, pred

    fgsm_proba, fgsm_pred = score_adv(fgsm_adv)
    pgd_proba, pgd_pred = score_adv(pgd_adv)

    attack_recall_clean = float((clean_pred[attack_mask] == 1).mean())
    attack_recall_fgsm = float((fgsm_pred == 1).mean())
    attack_recall_pgd = float((pgd_pred == 1).mean())

    anomaly_metrics = {}
    if anomaly_bundle is not None:
        fgsm_anomaly_score = anomaly_bundle["model"].decision_function(fgsm_adv)
        pgd_anomaly_score = anomaly_bundle["model"].decision_function(pgd_adv)
        anomaly_metrics = {
            "fgsm_mean_score": float(np.mean(fgsm_anomaly_score)),
            "pgd_mean_score": float(np.mean(pgd_anomaly_score)),
            "fgsm_anomaly_tpr": float(np.mean(fgsm_anomaly_score < 0)),
            "pgd_anomaly_tpr": float(np.mean(pgd_anomaly_score < 0)),
        }

    payload = {
        "classifier_bundle_path": str(classifier_bundle_path),
        "anomaly_bundle_path": str(anomaly_bundle_path) if anomaly_bundle_path else None,
        "attack_sample_count": int(len(attack_dense)),
        "epsilon": float(epsilon),
        "pgd_steps": int(pgd_steps),
        "pgd_step_size": float(pgd_step_size),
        "clean_metrics": {
            "attack_recall": attack_recall_clean,
            "attack_fnr": float(1.0 - attack_recall_clean),
            "confusion_matrix": clean_cm.tolist(),
            "pr_auc": float(average_precision_score(y_test, clean_proba)),
        },
        "fgsm_metrics": {
            "attack_recall": attack_recall_fgsm,
            "attack_fnr": float(1.0 - attack_recall_fgsm),
            "mean_attack_probability": float(np.mean(fgsm_proba)),
            "min_attack_probability": float(np.min(fgsm_proba)),
        },
        "pgd_metrics": {
            "attack_recall": attack_recall_pgd,
            "attack_fnr": float(1.0 - attack_recall_pgd),
            "mean_attack_probability": float(np.mean(pgd_proba)),
            "min_attack_probability": float(np.min(pgd_proba)),
        },
        "deltas": {
            "fgsm_recall_delta": float(attack_recall_fgsm - attack_recall_clean),
            "pgd_recall_delta": float(attack_recall_pgd - attack_recall_clean),
        },
        "anomaly_metrics": anomaly_metrics,
        "runtime": {
            "rows_total": int(len(X)),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "fit_seconds": 0.0,
        },
        "training_meta": training_meta,
    }
    write_json(output_path, payload)
    return payload
