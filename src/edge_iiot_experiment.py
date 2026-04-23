from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse as scipy_sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - optional dependency
    SMOTE = None


DEFAULT_EDGE_CSV = "data/ML-EdgeIIoT-dataset.csv"
DEFAULT_MODEL_PATH = "models/edge_iiot_xgb_model.joblib"
DEFAULT_REPORT_DIR = Path("output/reports")

LABEL_COLUMNS = {"Attack_label", "Attack_type"}

EDGE_DROP_IDENTITY_PAYLOAD_COLUMNS = {
    "frame.time",
    "ip.dst_host",
    "ip.src_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "icmp.transmit_timestamp",
    "http.request.uri.query",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "mqtt.msg",
}

MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "<nan>"}
CV_METRIC_COLUMNS = [
    "fold",
    "train_rows",
    "val_rows",
    "train_positive",
    "train_negative",
    "resampled_rows",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "attack_precision",
    "attack_recall",
    "attack_fnr",
    "tn",
    "fp",
    "fn",
    "tp",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df.loc[:, ~df.columns.duplicated()]


def read_csv(path: str | Path, *, sample_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False, nrows=sample_rows)
    return normalize_columns(df)


def first_repeated_value(text: str) -> str:
    for sep in ("|", ";"):
        if sep in text:
            return text.split(sep, 1)[0].strip()
    return text


def parse_numeric_value(value: object) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return np.nan
    text = first_repeated_value(text)

    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return 1.0
    if lowered in {"false", "no"}:
        return 0.0

    if re.fullmatch(r"0x[0-9a-fA-F]+", text):
        return float(int(text, 16))

    if "," in text:
        return np.nan

    return float(pd.to_numeric(text, errors="coerce"))


def numeric_parse_ratio(series: pd.Series) -> tuple[pd.Series, float, int]:
    raw = series.astype("string")
    non_empty = raw.notna() & ~raw.str.strip().str.lower().isin(MISSING_STRINGS)
    parsed = series.map(parse_numeric_value)
    denominator = int(non_empty.sum())
    if denominator == 0:
        return parsed, 1.0, 0
    return parsed, float(parsed.notna().sum() / denominator), denominator


def clean_string_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("__MISSING__").str.strip()
    return values.mask(values.str.lower().isin(MISSING_STRINGS), "__MISSING__").astype(str)


def coerce_feature_types(
    df: pd.DataFrame,
    *,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    numeric_threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, object]]:
    df = normalize_columns(df)
    out = pd.DataFrame(index=df.index)

    if numeric_columns is not None or categorical_columns is not None:
        numeric_columns = numeric_columns or []
        categorical_columns = categorical_columns or []
        for col in numeric_columns:
            out[col] = df[col].map(parse_numeric_value) if col in df.columns else np.nan
        for col in categorical_columns:
            out[col] = clean_string_series(df[col]) if col in df.columns else "__MISSING__"
        return out, numeric_columns, categorical_columns, {
            "numeric_parse_ratios": {},
            "numeric_non_empty_counts": {},
        }

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    ratios: dict[str, float] = {}
    non_empty_counts: dict[str, int] = {}

    for col in df.columns:
        parsed, ratio, non_empty_count = numeric_parse_ratio(df[col])
        ratios[col] = ratio
        non_empty_counts[col] = non_empty_count
        if ratio >= numeric_threshold:
            out[col] = parsed
            numeric_cols.append(col)
        else:
            out[col] = clean_string_series(df[col])
            categorical_cols.append(col)

    return out, numeric_cols, categorical_cols, {
        "numeric_parse_ratios": ratios,
        "numeric_non_empty_counts": non_empty_counts,
    }


def build_binary_labels(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "Attack_label" in df.columns:
        parsed = df["Attack_label"].map(parse_numeric_value)
        if parsed.notna().any():
            return (parsed.fillna(0) > 0).astype(int), "Attack_label"

        text = df["Attack_label"].astype(str).str.strip().str.lower()
        return (~text.isin({"0", "normal", "benign", "false"})).astype(int), "Attack_label"

    if "Attack_type" in df.columns:
        text = df["Attack_type"].astype(str).str.strip().str.lower()
        return (~text.isin({"normal", "benign", "0", "false"})).astype(int), "Attack_type"

    raise ValueError("No label column found. Expected Attack_label or Attack_type.")


def make_one_hot_encoder(min_frequency: int) -> OneHotEncoder:
    attempts = []
    base_kwargs = {"handle_unknown": "ignore"}
    if min_frequency > 1:
        base_kwargs["min_frequency"] = min_frequency

    for sparse_key in ("sparse_output", "sparse"):
        kwargs = dict(base_kwargs)
        kwargs[sparse_key] = True
        attempts.append(kwargs)

    for kwargs in attempts:
        try:
            return OneHotEncoder(**kwargs)
        except TypeError:
            continue

    return OneHotEncoder(handle_unknown="ignore")


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    min_category_count: int,
) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", make_one_hot_encoder(min_category_count)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns remain after preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)


def train_xgb(X_train, y_train: pd.Series) -> XGBClassifier:
    return train_xgb_with_balance(X_train, y_train, balanced_training=False)


def train_xgb_with_balance(
    X_train,
    y_train: pd.Series,
    *,
    balanced_training: bool,
) -> XGBClassifier:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=350,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=1.0 if balanced_training else ((neg / pos) if pos else 1.0),
    )
    model.fit(X_train, y_train)
    return model


def choose_threshold(
    y_true: pd.Series,
    pred_proba: np.ndarray,
    *,
    strategy: str,
    fixed_threshold: float,
    min_precision: float,
) -> tuple[float, dict[str, float]]:
    if strategy == "fixed":
        return fixed_threshold, {}

    precision, recall, thresholds = precision_recall_curve(y_true, pred_proba)
    if len(thresholds) == 0:
        return fixed_threshold, {}

    precision = precision[:-1]
    recall = recall[:-1]
    if strategy == "f1":
        scores = (2 * precision * recall) / (precision + recall + 1e-12)
    elif strategy == "f2":
        scores = (5 * precision * recall) / (4 * precision + recall + 1e-12)
    else:
        raise ValueError(f"Unsupported threshold strategy: {strategy}")

    if min_precision > 0 and (precision >= min_precision).any():
        scores = np.where(precision >= min_precision, scores, -np.inf)

    idx = int(np.argmax(scores))
    return float(thresholds[idx]), {
        "validation_precision": float(precision[idx]),
        "validation_recall": float(recall[idx]),
        "validation_score": float(scores[idx]),
    }


def evaluation_from_predictions(y_true: pd.Series, pred_proba: np.ndarray, *, threshold: float) -> dict[str, object]:
    pred_label = (pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred_label, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    per_class_report = classification_report(
        y_true,
        pred_label,
        target_names=["Normal", "Attack"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "roc_auc": float(roc_auc_score(y_true, pred_proba)),
        "pr_auc": float(average_precision_score(y_true, pred_proba)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": per_class_report,
    }

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "pred_label": pred_label,
        "classification_report": per_class_report,
    }


def resample_with_smote(X_train_tx, y_train: pd.Series) -> tuple[object, pd.Series]:
    if SMOTE is None:
        raise RuntimeError(
            "imbalanced-learn is required for --use_smote. Install it with `pip install imbalanced-learn`."
        )

    y_series = pd.Series(y_train).reset_index(drop=True)
    class_counts = y_series.value_counts().sort_index()
    minority_count = int(class_counts.min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least two samples in the minority class within each fold.")

    if scipy_sparse.issparse(X_train_tx):
        X_train_tx = X_train_tx.tocsr()

    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train_tx, y_series)
    return X_resampled, pd.Series(y_resampled)


def run_smote_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int,
    min_category_count: int,
    threshold: float,
    training_meta: dict[str, object],
) -> dict[str, object]:
    if SMOTE is None:
        raise RuntimeError(
            "imbalanced-learn is required for SMOTE cross-validation. Install it with `pip install imbalanced-learn`."
        )
    if cv_folds < 2:
        raise ValueError("--cv_folds must be at least 2.")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_rows: list[dict[str, object]] = []
    fold_predictions: list[pd.DataFrame] = []
    fold_confusions: list[dict[str, int]] = []
    overall_cm = np.zeros((2, 2), dtype=int)

    start = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        preprocessor = make_preprocessor(
            training_meta["numeric_columns"],
            training_meta["categorical_columns"],
            min_category_count=min_category_count,
        )
        X_train_tx = preprocessor.fit_transform(X_train)
        X_val_tx = preprocessor.transform(X_val)

        X_resampled, y_resampled = resample_with_smote(X_train_tx, y_train)
        model = train_xgb_with_balance(X_resampled, y_resampled, balanced_training=True)
        val_proba = model.predict_proba(X_val_tx)[:, 1]
        fold_eval = evaluation_from_predictions(y_val, val_proba, threshold=threshold)
        metrics = fold_eval["metrics"]
        cm = fold_eval["confusion_matrix"]
        overall_cm += cm
        fold_confusions.append(
            {
                "fold": fold,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "train_positive": int((y_train == 1).sum()),
                "train_negative": int((y_train == 0).sum()),
                "resampled_rows": int(len(y_resampled)),
                "accuracy": float(metrics["accuracy"]),
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "attack_precision": float(metrics["precision"]),
                "attack_recall": float(metrics["recall"]),
                "attack_fnr": float(metrics["fnr"]),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )
        fold_predictions.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "record_index": val_idx,
                    "true_label": y_val.values,
                    "pred_proba_attack": val_proba,
                    "pred_label": (val_proba >= threshold).astype(int),
                }
            )
        )

    fold_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(fold_predictions, ignore_index=True).sort_values(
        ["fold", "record_index"]
    )
    overall_eval = evaluation_from_predictions(
        predictions_df["true_label"],
        predictions_df["pred_proba_attack"],
        threshold=threshold,
    )

    metric_summary = {}
    for column in [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "attack_precision",
        "attack_recall",
        "attack_fnr",
    ]:
        metric_summary[column] = {
            "mean": float(fold_df[column].mean()),
            "std": float(fold_df[column].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        }

    summary = {
        "cv_folds": int(cv_folds),
        "threshold": float(threshold),
        "use_smote": True,
        "smote_training_only": True,
        "strategy": "stratified_kfold_with_fold_local_smote",
        "metric_summary": metric_summary,
        "overall_oof_metrics": {
            "accuracy": float(overall_eval["metrics"]["accuracy"]),
            "roc_auc": float(overall_eval["metrics"]["roc_auc"]),
            "pr_auc": float(overall_eval["metrics"]["pr_auc"]),
            "attack_precision": float(overall_eval["metrics"]["precision"]),
            "attack_recall": float(overall_eval["metrics"]["recall"]),
            "attack_fnr": float(overall_eval["metrics"]["fnr"]),
        },
        "aggregate_confusion_matrix": overall_cm.tolist(),
        "fold_confusions": fold_confusions,
        "rows_total": int(len(X)),
        "runtime_seconds": float(time.perf_counter() - start),
    }

    return {
        "fold_metrics": fold_df,
        "predictions": predictions_df,
        "summary": summary,
        "overall_confusion_matrix": overall_cm,
        "overall_eval": overall_eval,
    }


def prepare_training_frame(
    edge_csv: str | Path,
    *,
    keep_identity_payload: bool,
    numeric_threshold: float,
    sample_rows: int | None,
    drop_duplicates: bool,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    df = read_csv(edge_csv, sample_rows=sample_rows)
    original_rows = len(df)
    if drop_duplicates:
        df = df.drop_duplicates()

    y, label_source = build_binary_labels(df)
    raw_feature_columns = [col for col in df.columns if col not in LABEL_COLUMNS]
    dropped_columns = []
    if not keep_identity_payload:
        dropped_columns = [col for col in raw_feature_columns if col in EDGE_DROP_IDENTITY_PAYLOAD_COLUMNS]
        raw_feature_columns = [col for col in raw_feature_columns if col not in EDGE_DROP_IDENTITY_PAYLOAD_COLUMNS]

    X_raw = df[raw_feature_columns].copy()
    X_typed, numeric_columns, categorical_columns, diagnostics = coerce_feature_types(
        X_raw,
        numeric_threshold=numeric_threshold,
    )

    empty_columns = [col for col in X_typed.columns if X_typed[col].isna().all() or (X_typed[col] == "__MISSING__").all()]
    if empty_columns:
        X_typed = X_typed.drop(columns=empty_columns)
        numeric_columns = [col for col in numeric_columns if col not in empty_columns]
        categorical_columns = [col for col in categorical_columns if col not in empty_columns]

    constant_columns = []
    for col in X_typed.columns:
        if X_typed[col].nunique(dropna=True) <= 1:
            constant_columns.append(col)
    if constant_columns:
        X_typed = X_typed.drop(columns=constant_columns)
        numeric_columns = [col for col in numeric_columns if col not in constant_columns]
        categorical_columns = [col for col in categorical_columns if col not in constant_columns]

    if X_typed.empty:
        raise ValueError("No usable feature columns remain after preprocessing.")

    training_meta = {
        "edge_csv": str(edge_csv),
        "original_rows": int(original_rows),
        "rows_after_drop_duplicates": int(len(df)),
        "label_source": label_source,
        "raw_edge_columns": list(df.columns),
        "raw_feature_columns_before_drops": [col for col in df.columns if col not in LABEL_COLUMNS],
        "dropped_identity_payload_columns": dropped_columns,
        "dropped_empty_columns": empty_columns,
        "dropped_constant_columns": constant_columns,
        "feature_columns": list(X_typed.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_parse_ratios": diagnostics["numeric_parse_ratios"],
        "numeric_non_empty_counts": diagnostics["numeric_non_empty_counts"],
    }
    return X_typed, y.reset_index(drop=True), training_meta


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names: list[str] = []
        for name, _, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            names.extend([str(col) for col in columns])
        return names


def fit_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    min_category_count: int,
    threshold_strategy: str,
    fixed_threshold: float,
    min_precision: float,
    training_meta: dict[str, object],
) -> dict[str, object]:
    fit_started = time.perf_counter()
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.20,
        random_state=43,
        stratify=y.iloc[train_idx],
    )

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    preprocessor = make_preprocessor(
        training_meta["numeric_columns"],
        training_meta["categorical_columns"],
        min_category_count=min_category_count,
    )
    threshold_fit_started = time.perf_counter()
    X_train_tx = preprocessor.fit_transform(X_train)
    threshold_model = train_xgb(X_train_tx, y_train)
    threshold_fit_seconds = time.perf_counter() - threshold_fit_started

    validation_predict_started = time.perf_counter()
    val_proba = threshold_model.predict_proba(preprocessor.transform(X_val))[:, 1]
    validation_predict_seconds = time.perf_counter() - validation_predict_started
    threshold, threshold_meta = choose_threshold(
        y_val,
        val_proba,
        strategy=threshold_strategy,
        fixed_threshold=fixed_threshold,
        min_precision=min_precision,
    )

    print("\nThreshold selection:")
    print(f"Strategy           : {threshold_strategy}")
    print(f"Selected threshold  : {threshold:.9f}")
    for key, value in threshold_meta.items():
        print(f"{key:18s}: {value:.6f}")

    dev_idx = np.concatenate([train_idx, val_idx])
    eval_preprocessor = make_preprocessor(
        training_meta["numeric_columns"],
        training_meta["categorical_columns"],
        min_category_count=min_category_count,
    )
    eval_fit_started = time.perf_counter()
    X_dev_tx = eval_preprocessor.fit_transform(X.iloc[dev_idx])
    eval_model = train_xgb(X_dev_tx, y.iloc[dev_idx])
    eval_fit_seconds = time.perf_counter() - eval_fit_started

    test_predict_started = time.perf_counter()
    test_proba = eval_model.predict_proba(eval_preprocessor.transform(X_test))[:, 1]
    test_predict_seconds = time.perf_counter() - test_predict_started
    test_eval = evaluation_from_predictions(y_test, test_proba, threshold=threshold)
    metrics = test_eval["metrics"]
    cm = test_eval["confusion_matrix"]
    pred_label = test_eval["pred_label"]
    report = test_eval["classification_report"]

    final_preprocessor = make_preprocessor(
        training_meta["numeric_columns"],
        training_meta["categorical_columns"],
        min_category_count=min_category_count,
    )
    final_fit_started = time.perf_counter()
    X_full_tx = final_preprocessor.fit_transform(X)
    final_model = train_xgb(X_full_tx, y)
    final_fit_seconds = time.perf_counter() - final_fit_started

    transformed_feature_names = get_transformed_feature_names(final_preprocessor)
    importance = pd.DataFrame(
        {
            "feature": transformed_feature_names[: len(final_model.feature_importances_)],
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    runtime = {
        "rows_total": int(len(X)),
        "features_total": int(X.shape[1]),
        "train_rows_threshold_model": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "threshold_model_fit_seconds": float(threshold_fit_seconds),
        "validation_predict_seconds": float(validation_predict_seconds),
        "eval_model_fit_seconds": float(eval_fit_seconds),
        "test_predict_seconds": float(test_predict_seconds),
        "test_predict_rows_per_second": float(len(X_test) / test_predict_seconds)
        if test_predict_seconds > 0
        else float("inf"),
        "final_model_fit_seconds": float(final_fit_seconds),
        "fit_bundle_seconds": float(time.perf_counter() - fit_started),
    }

    print("\nHoldout evaluation:")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC     : {metrics['pr_auc']:.4f}")
    print(f"Attack P   : {metrics['precision']:.4f}")
    print(f"Attack R   : {metrics['recall']:.4f}")
    print(f"Attack FNR : {metrics['fnr']:.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]")
    print(cm)
    print("\nClassification report:")
    print(pd.DataFrame(report).T.to_string())

    holdout_predictions = pd.DataFrame(
        {
            "true_label": y_test.values,
            "pred_proba_attack": test_proba,
            "pred_label": pred_label,
        }
    )
    holdout_predictions["true_label_name"] = holdout_predictions["true_label"].map({0: "Normal", 1: "Attack"})
    holdout_predictions["pred_label_name"] = holdout_predictions["pred_label"].map({0: "Normal", 1: "Attack"})

    return {
        "model": final_model,
        "preprocessor": final_preprocessor,
        "threshold": threshold,
        "threshold_strategy": threshold_strategy,
        "fixed_threshold": fixed_threshold,
        "threshold_meta": threshold_meta,
        "training_meta": training_meta,
        "evaluation_metrics": metrics,
        "evaluation_report": report,
        "confusion_matrix": cm,
        "holdout_predictions": holdout_predictions,
        "runtime": runtime,
        "feature_importance": importance,
        "threshold_model_fit_seconds": threshold_fit_seconds,
        "eval_model_fit_seconds": eval_fit_seconds,
        "final_model_fit_seconds": final_fit_seconds,
    }


def save_reports(bundle: dict[str, object], report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = report_dir / "edge_iiot_holdout_metrics.json"
    predictions_path = report_dir / "edge_iiot_holdout_predictions.csv"
    confusion_matrix_path = report_dir / "edge_iiot_holdout_confusion_matrix.csv"
    class_report_path = report_dir / "edge_iiot_classification_report.csv"

    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "threshold": bundle["threshold"],
                "threshold_strategy": bundle["threshold_strategy"],
                "fixed_threshold": bundle["fixed_threshold"],
                "threshold_meta": bundle["threshold_meta"],
                "training_meta": bundle["training_meta"],
                "evaluation_metrics": bundle["evaluation_metrics"],
                "runtime": bundle["runtime"],
            },
            fh,
            indent=2,
        )

    bundle["holdout_predictions"].to_csv(predictions_path, index=False)
    pd.DataFrame(
        bundle["confusion_matrix"],
        index=["true_normal", "true_attack"],
        columns=["pred_normal", "pred_attack"],
    ).to_csv(confusion_matrix_path)
    pd.DataFrame(bundle["evaluation_report"]).T.to_csv(class_report_path)

    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "confusion_matrix": confusion_matrix_path,
        "classification_report": class_report_path,
    }


def save_cv_reports(cv_result: dict[str, object], report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics_path = report_dir / "edge_iiot_cv_fold_metrics.csv"
    predictions_path = report_dir / "edge_iiot_cv_predictions.csv"
    summary_path = report_dir / "edge_iiot_cv_summary.json"
    confusion_matrix_path = report_dir / "edge_iiot_cv_confusion_matrix.csv"

    cv_result["fold_metrics"].to_csv(fold_metrics_path, index=False)
    cv_result["predictions"].to_csv(predictions_path, index=False)
    pd.DataFrame(
        cv_result["overall_confusion_matrix"],
        index=["true_normal", "true_attack"],
        columns=["pred_normal", "pred_attack"],
    ).to_csv(confusion_matrix_path)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(cv_result["summary"], fh, indent=2)

    return {
        "fold_metrics": fold_metrics_path,
        "predictions": predictions_path,
        "summary": summary_path,
        "confusion_matrix": confusion_matrix_path,
    }


def train_command(args: argparse.Namespace) -> None:
    command_started = time.perf_counter()
    prep_started = time.perf_counter()
    X, y, training_meta = prepare_training_frame(
        args.edge_csv,
        keep_identity_payload=args.keep_identity_payload,
        numeric_threshold=args.numeric_threshold,
        sample_rows=args.sample_rows,
        drop_duplicates=not args.keep_duplicates,
    )
    preprocessing_seconds = time.perf_counter() - prep_started

    print("\nTraining data:")
    print(f"Rows                   : {len(X):,}")
    print(f"Features used          : {X.shape[1]:,}")
    print(f"Numeric features       : {len(training_meta['numeric_columns']):,}")
    print(f"Categorical features   : {len(training_meta['categorical_columns']):,}")
    print(f"Dropped ID/payload cols: {len(training_meta['dropped_identity_payload_columns']):,}")
    print(f"Dropped empty cols     : {len(training_meta['dropped_empty_columns']):,}")
    print(f"Dropped constant cols  : {len(training_meta['dropped_constant_columns']):,}")
    print("Class distribution:")
    print(y.value_counts().sort_index().rename(index={0: "normal", 1: "attack"}).to_string())

    bundle = fit_bundle(
        X,
        y,
        min_category_count=args.min_category_count,
        threshold_strategy=args.threshold_strategy,
        fixed_threshold=args.fixed_threshold,
        min_precision=args.min_precision,
        training_meta=training_meta,
    )
    total_seconds = time.perf_counter() - command_started
    bundle["runtime"]["preprocessing_seconds"] = float(preprocessing_seconds)
    bundle["runtime"]["train_command_total_seconds"] = float(total_seconds)

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle["model"],
            "preprocessor": bundle["preprocessor"],
            "threshold": bundle["threshold"],
            "threshold_strategy": bundle["threshold_strategy"],
            "fixed_threshold": bundle["fixed_threshold"],
            "threshold_meta": bundle["threshold_meta"],
            "training_meta": bundle["training_meta"],
            "evaluation_metrics": bundle["evaluation_metrics"],
            "evaluation_report": bundle["evaluation_report"],
            "runtime": bundle["runtime"],
        },
        model_path,
    )
    print(f"\nSaved model bundle: {model_path}")

    importance_path = model_path.with_suffix(".feature_importance.csv")
    bundle["feature_importance"].to_csv(importance_path, index=False)
    print(f"Saved feature importance: {importance_path}")

    meta_path = model_path.with_suffix(".metadata.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "threshold": bundle["threshold"],
                "threshold_strategy": bundle["threshold_strategy"],
                "threshold_meta": bundle["threshold_meta"],
                "training_meta": bundle["training_meta"],
                "evaluation_metrics": bundle["evaluation_metrics"],
                "runtime": bundle["runtime"],
            },
            fh,
            indent=2,
        )
    print(f"Saved metadata: {meta_path}")

    report_paths = save_reports(bundle, DEFAULT_REPORT_DIR)
    print("Saved evaluation reports:")
    for label, path in report_paths.items():
        print(f"  {label}: {path}")

    if args.use_smote:
        print("\nRunning SMOTE + stratified cross-validation on training folds only...")
        cv_result = run_smote_cross_validation(
            X,
            y,
            cv_folds=args.cv_folds,
            min_category_count=args.min_category_count,
            threshold=args.fixed_threshold,
            training_meta=training_meta,
        )
        cv_paths = save_cv_reports(cv_result, DEFAULT_REPORT_DIR)
        print("Saved CV reports:")
        for label, path in cv_paths.items():
            print(f"  {label}: {path}")

        summary = cv_result["summary"]
        print("\nSMOTE + CV summary:")
        for metric, stats in summary["metric_summary"].items():
            print(f"{metric:18s} mean={stats['mean']:.4f} std={stats['std']:.4f}")
        print("Out-of-fold metrics:")
        for metric, value in summary["overall_oof_metrics"].items():
            print(f"{metric:18s} {value:.4f}")
        print("Aggregate confusion matrix [ [TN, FP], [FN, TP] ]")
        print(pd.DataFrame(
            summary["aggregate_confusion_matrix"],
            index=["true_normal", "true_attack"],
            columns=["pred_normal", "pred_attack"],
        ).to_string())

    print(f"Total train command: {total_seconds:.3f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Edge-IIoT binary IDS trainer using the archived XGBoost pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the offline Edge-IIoT binary classifier.")
    train_parser.add_argument("--edge_csv", default=DEFAULT_EDGE_CSV)
    train_parser.add_argument("--model_out", default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--sample_rows", type=int, default=None, help="Optional row limit for quick tests.")
    train_parser.add_argument("--keep_duplicates", action="store_true", help="Do not drop duplicate rows.")
    train_parser.add_argument(
        "--keep_identity_payload",
        action="store_true",
        help="Keep timestamp/IP/port/payload columns. Default drops them to reduce memorization.",
    )
    train_parser.add_argument("--numeric_threshold", type=float, default=0.95)
    train_parser.add_argument("--min_category_count", type=int, default=20)
    train_parser.add_argument(
        "--threshold_strategy",
        choices=["fixed", "f1", "f2"],
        default="fixed",
        help="Use the fixed 0.5 threshold by default; optional validation tuning uses f1/f2.",
    )
    train_parser.add_argument("--fixed_threshold", type=float, default=0.5)
    train_parser.add_argument("--min_precision", type=float, default=0.0)
    train_parser.add_argument(
        "--use_smote",
        action="store_true",
        help="Run additional SMOTE + stratified CV evaluation on training folds only.",
    )
    train_parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of stratified folds to use when --use_smote is enabled.",
    )
    train_parser.set_defaults(func=train_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "train":
        raise SystemExit("Only the train command is supported in this offline rebuild.")

    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
