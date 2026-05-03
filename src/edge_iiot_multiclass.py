from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from edge_iiot_experiment import (
    DEFAULT_EDGE_CSV,
    EDGE_DROP_IDENTITY_PAYLOAD_COLUMNS,
    LABEL_COLUMNS,
    build_binary_labels,
    coerce_feature_types,
    get_transformed_feature_names,
    make_preprocessor,
    normalize_columns,
    read_csv,
)
from edge_iiot_runtime import json_safe, save_feature_contract, save_active_model_pointer, utc_now


NORMAL_TOKENS = {"normal", "benign", "0", "false"}


def _normalize_attack_type(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in NORMAL_TOKENS:
        return None
    return text


def prepare_attack_training_frame(
    edge_csv: str | Path = DEFAULT_EDGE_CSV,
    *,
    keep_identity_payload: bool = False,
    numeric_threshold: float = 0.95,
    sample_rows: int | None = None,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, pd.Series, LabelEncoder, dict[str, object]]:
    df = read_csv(edge_csv, sample_rows=sample_rows)
    original_rows = len(df)
    if drop_duplicates:
        df = df.drop_duplicates()

    if "Attack_type" not in df.columns:
        raise ValueError("No Attack_type column found for multiclass attack training.")

    attack_labels = df["Attack_type"].map(_normalize_attack_type)
    attack_mask = attack_labels.notna()
    attack_df = df.loc[attack_mask].copy()
    attack_labels = attack_labels.loc[attack_mask].astype(str).reset_index(drop=True)
    attack_df = attack_df.reset_index(drop=True)

    raw_feature_columns = [col for col in attack_df.columns if col not in LABEL_COLUMNS]
    dropped_columns: list[str] = []
    if not keep_identity_payload:
        dropped_columns = [col for col in raw_feature_columns if col in EDGE_DROP_IDENTITY_PAYLOAD_COLUMNS]
        raw_feature_columns = [col for col in raw_feature_columns if col not in EDGE_DROP_IDENTITY_PAYLOAD_COLUMNS]

    X_raw = attack_df[raw_feature_columns].copy()
    X_typed, numeric_columns, categorical_columns, diagnostics = coerce_feature_types(
        X_raw,
        numeric_threshold=numeric_threshold,
    )

    empty_columns = [col for col in X_typed.columns if X_typed[col].isna().all() or (X_typed[col] == "__MISSING__").all()]
    if empty_columns:
        X_typed = X_typed.drop(columns=empty_columns)
        numeric_columns = [col for col in numeric_columns if col not in empty_columns]
        categorical_columns = [col for col in categorical_columns if col not in empty_columns]

    constant_columns: list[str] = []
    for col in X_typed.columns:
        if X_typed[col].nunique(dropna=True) <= 1:
            constant_columns.append(col)
    if constant_columns:
        X_typed = X_typed.drop(columns=constant_columns)
        numeric_columns = [col for col in numeric_columns if col not in constant_columns]
        categorical_columns = [col for col in categorical_columns if col not in constant_columns]

    if X_typed.empty:
        raise ValueError("No usable multiclass features remain after preprocessing.")

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(attack_labels), name="Attack_type_index")
    training_meta = {
        "edge_csv": str(edge_csv),
        "original_rows": int(original_rows),
        "rows_after_drop_duplicates": int(len(df)),
        "attack_rows": int(len(attack_df)),
        "label_source": "Attack_type",
        "attack_classes": list(label_encoder.classes_),
        "raw_edge_columns": list(attack_df.columns),
        "raw_feature_columns_before_drops": [col for col in attack_df.columns if col not in LABEL_COLUMNS],
        "dropped_identity_payload_columns": dropped_columns,
        "dropped_empty_columns": empty_columns,
        "dropped_constant_columns": constant_columns,
        "feature_columns": list(X_typed.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_parse_ratios": diagnostics["numeric_parse_ratios"],
        "numeric_non_empty_counts": diagnostics["numeric_non_empty_counts"],
    }
    return X_typed, y.reset_index(drop=True), label_encoder, training_meta


def train_attack_model(
    X_train_tx,
    y_train: pd.Series,
    *,
    class_count: int,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    params = params or {}
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=class_count,
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.85)),
        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        min_child_weight=int(params.get("min_child_weight", 1)),
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train_tx, y_train, sample_weight=sample_weight)
    return model


def evaluate_attack_model(
    y_true: pd.Series,
    proba: np.ndarray,
    pred_idx: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    pred_names = [classes[int(idx)] for idx in pred_idx]
    report = classification_report(
        y_true,
        pred_idx,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, pred_idx, labels=list(range(len(classes))))

    per_class_pr_auc: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    for idx, class_name in enumerate(classes):
        binary_true = (y_true == idx).astype(int)
        if len(np.unique(binary_true)) < 2:
            per_class_pr_auc[class_name] = 0.0
            per_class_recall[class_name] = 0.0
            continue
        per_class_pr_auc[class_name] = float(average_precision_score(binary_true, proba[:, idx]))
        per_class_recall[class_name] = float(report[class_name]["recall"])

    return {
        "pred_label_index": pred_idx,
        "pred_label_name": pred_names,
        "classification_report": report,
        "confusion_matrix": cm,
        "metrics": {
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(f1_score(y_true, pred_idx, average="macro")),
            "weighted_f1": float(f1_score(y_true, pred_idx, average="weighted")),
            "per_class_recall": per_class_recall,
            "per_class_pr_auc": per_class_pr_auc,
        },
    }


def threshold_summary_for_class(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    *,
    default_threshold: float = 0.5,
    min_precision: float = 0.97,
    critical: bool = False,
) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)
    if len(thresholds) == 0:
        return {
            "threshold": default_threshold,
            "strategy": "default",
            "precision": 0.0,
            "recall": 0.0,
            "fnr": 1.0,
        }
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
    f2 = (5.0 * precision * recall) / (4.0 * precision + recall + 1e-12)
    score = f2 if critical else f1
    feasible_mask = precision >= min_precision
    if feasible_mask.any():
        feasible_score = np.where(feasible_mask, score, -np.inf)
    else:
        feasible_score = score
    idx = int(np.argmax(feasible_score))
    threshold = float(thresholds[idx])
    return {
        "threshold": threshold,
        "strategy": "f2" if critical else "f1",
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "fnr": float(1.0 - recall[idx]),
        "f1": float(f1[idx]),
        "f2": float(f2[idx]),
        "min_precision_constraint": float(min_precision),
        "precision_constraint_met": bool(feasible_mask.any()),
        "default_threshold": float(default_threshold),
    }


def calibrate_multiclass_thresholds(
    y_true: pd.Series,
    proba: np.ndarray,
    classes: list[str],
    *,
    default_threshold: float = 0.5,
    min_precision: float = 0.97,
    critical_classes: Iterable[str] | None = None,
) -> dict[str, Any]:
    critical_classes = set(critical_classes or [])
    rows: list[dict[str, Any]] = []
    for idx, class_name in enumerate(classes):
        y_binary = (y_true == idx).astype(int).to_numpy()
        summary = threshold_summary_for_class(
            y_binary,
            proba[:, idx],
            default_threshold=default_threshold,
            min_precision=min_precision,
            critical=class_name in critical_classes,
        )
        summary["class_name"] = class_name
        summary["class_index"] = idx
        summary["critical"] = class_name in critical_classes
        rows.append(summary)
    df = pd.DataFrame(rows).sort_values(["critical", "recall", "precision"], ascending=[False, False, False])
    return {
        "default_threshold": float(default_threshold),
        "min_precision": float(min_precision),
        "critical_classes": sorted(critical_classes),
        "per_class": rows,
        "table": df,
    }


def train_attack_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    label_encoder: LabelEncoder,
    training_meta: dict[str, Any],
    min_category_count: int,
    random_state: int = 42,
    threshold_min_precision: float = 0.97,
    critical_classes: Iterable[str] | None = None,
    model_out: str | Path = "models/edge_iiot_attack_xgb_model.joblib",
    feature_contract_out: str | Path = "models/edge_iiot_feature_contract.json",
    pointer_out: str | Path = "models/edge_iiot_active_model.json",
) -> dict[str, Any]:
    start = time.perf_counter()
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    preprocessor = make_preprocessor(
        training_meta["numeric_columns"],
        training_meta["categorical_columns"],
        min_category_count=min_category_count,
    )
    X_train_tx = preprocessor.fit_transform(X_train)
    X_test_tx = preprocessor.transform(X_test)
    model = train_attack_model(X_train_tx, y_train, class_count=len(label_encoder.classes_), random_state=random_state)
    proba = model.predict_proba(X_test_tx)
    pred_idx = np.argmax(proba, axis=1)
    eval_result = evaluate_attack_model(y_test, proba, pred_idx, list(label_encoder.classes_))
    thresholds = calibrate_multiclass_thresholds(
        y_test,
        proba,
        list(label_encoder.classes_),
        default_threshold=0.5,
        min_precision=threshold_min_precision,
        critical_classes=critical_classes,
    )

    transformed_feature_names = get_transformed_feature_names(preprocessor)
    importance = pd.DataFrame(
        {
            "feature": transformed_feature_names[: len(model.feature_importances_)],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    holdout_predictions = pd.DataFrame(
        {
            "dataset_row_index": test_idx,
            "true_label_index": y_test.to_numpy(),
            "true_label_name": label_encoder.inverse_transform(y_test.to_numpy()),
            "pred_label_index": pred_idx,
            "pred_label_name": [label_encoder.classes_[int(idx)] for idx in pred_idx],
        }
    )
    for idx, class_name in enumerate(label_encoder.classes_):
        holdout_predictions[f"proba_{class_name}"] = proba[:, idx]
    holdout_predictions["pred_proba_max"] = proba.max(axis=1)
    holdout_predictions["pred_rank"] = np.argmax(proba, axis=1)

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "classes_": list(label_encoder.classes_),
        "threshold": 0.5,
        "threshold_strategy": "multiclass_top1",
        "threshold_meta": json_safe({**thresholds, "table": thresholds["table"].to_dict(orient="records")}),
        "training_meta": training_meta,
        "evaluation_metrics": eval_result["metrics"],
        "evaluation_report": eval_result["classification_report"],
        "confusion_matrix": eval_result["confusion_matrix"],
        "holdout_predictions": holdout_predictions,
        "feature_importance": importance,
        "runtime": {
            "rows_total": int(len(X)),
            "attack_rows": int(len(y)),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "fit_seconds": float(time.perf_counter() - start),
        },
    }
    joblib.dump(bundle, model_path)

    feature_contract_path = save_feature_contract(
        training_meta,
        feature_contract_out,
        dataset_name=str(training_meta.get("edge_csv", DEFAULT_EDGE_CSV)),
        model_family="attack_multiclass_xgb",
        label_source="Attack_type",
        extra={
            "attack_classes": list(label_encoder.classes_),
            "transformed_feature_names": transformed_feature_names,
        },
    )
    pointer_path = save_active_model_pointer(
        pointer_out,
        binary_model_path=training_meta.get("binary_model_path", "models/edge_iiot_xgb_model.joblib"),
        attack_model_path=model_path,
        feature_contract_path=feature_contract_path,
        multiclass_threshold_path=Path("output/reports/edge_iiot_multiclass_thresholds.json"),
        version=training_meta.get("version", None),
        extra={
            "kind": "active_model_pointer",
            "attack_classes": list(label_encoder.classes_),
        },
    )

    return {
        "bundle": bundle,
        "bundle_path": model_path,
        "feature_contract_path": feature_contract_path,
        "pointer_path": pointer_path,
        "thresholds": thresholds,
    }
