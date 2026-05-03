from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from edge_iiot_drift import compare_batches, json_safe
from edge_iiot_experiment import (
    DEFAULT_EDGE_CSV,
    evaluation_from_predictions,
    get_transformed_feature_names,
    make_preprocessor,
    prepare_training_frame,
    coerce_feature_types,
    train_xgb_with_balance,
)
from edge_iiot_mongo import PREDICTIONS_COLLECTION, RETRAIN_EVENTS_COLLECTION, collection_to_dataframe, connect_database, insert_documents
from edge_iiot_runtime import (
    DEFAULT_ACTIVE_MODEL_POINTER_PATH,
    attach_model_feature_names,
    load_active_model_pointer,
    resolve_model_path,
    save_active_model_pointer,
    validate_runtime_contract,
)


DEFAULT_CLASSIFIER_BUNDLE = Path("models/edge_iiot_xgb_model.joblib")
DEFAULT_RETRAINED_BUNDLE = Path("models/edge_iiot_xgb_model_retrained.joblib")
DEFAULT_REPORT_DIR = Path("output/reports")
DEFAULT_TRIGGER_JSON = DEFAULT_REPORT_DIR / "edge_iiot_retrain_trigger.json"
DEFAULT_COMPARISON_JSON = DEFAULT_REPORT_DIR / "edge_iiot_retrain_comparison.json"
DEFAULT_COMPARISON_CSV = DEFAULT_REPORT_DIR / "edge_iiot_retrain_comparison.csv"
DEFAULT_RUN_SUMMARY = DEFAULT_REPORT_DIR / "edge_iiot_retrain_run_summary.md"
DEFAULT_RECENT_LIVE_ROWS = 5000
DEFAULT_MODEL_POINTER_PATH = DEFAULT_ACTIVE_MODEL_POINTER_PATH

DEFAULT_REFERENCE_FRACTION = 0.60
DEFAULT_DRIFT_FRACTION = 0.20
DEFAULT_EVAL_FRACTION = 0.20
DEFAULT_HIGH_THRESHOLD = 0.25
DEFAULT_MODERATE_THRESHOLD = 0.10
DEFAULT_MODERATE_FEATURE_COUNT = 12
DEFAULT_MIN_RETRAIN_PR_AUC = 0.92
DEFAULT_MIN_RETRAIN_ATTACK_RECALL = 0.90
DEFAULT_MIN_RETRAIN_NORMAL_RECALL = 0.50


def load_bundle(model_path: Path) -> dict[str, object]:
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier bundle not found: {model_path}")
    if model_path.suffix.lower() == ".json":
        pointer = load_active_model_pointer(model_path)
        if not pointer:
            raise ValueError(f"Classifier pointer is missing or invalid: {model_path}")
        model_path = resolve_model_path(model_path, "binary_model_path")
    bundle = joblib.load(model_path)
    required = {"model", "preprocessor", "threshold", "training_meta"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Classifier bundle is missing required keys: {sorted(missing)}")
    attach_model_feature_names(bundle)
    return bundle


def load_recent_labeled_live_rows(mongo_uri: str, db_name: str, *, limit: int) -> pd.DataFrame:
    db = connect_database(mongo_uri, db_name)
    df = collection_to_dataframe(
        db[PREDICTIONS_COLLECTION],
        query={"kind": "live_prediction"},
        projection={"_id": 0},
        sort=[("created_at", -1), ("window_id", -1), ("record_index", -1)],
        limit=limit,
    )
    if df.empty:
        return df
    if "true_label" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")
    df = df[df["true_label"].notna()].copy()
    if df.empty:
        return df
    sort_columns = [column for column in ["created_at", "window_id", "record_index"] if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns, ascending=True, kind="mergesort")
    return df


def prepare_live_retraining_frame(
    live_df: pd.DataFrame,
    training_meta: dict[str, object],
    *,
    numeric_threshold: float,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = list(training_meta["feature_columns"])
    numeric_columns = list(training_meta.get("numeric_columns", []))
    categorical_columns = list(training_meta.get("categorical_columns", []))

    from edge_iiot_experiment import normalize_columns

    live_df = normalize_columns(live_df)
    work = pd.DataFrame(index=live_df.index)
    for column in feature_columns:
        if column in live_df.columns:
            work[column] = live_df[column]
        elif column in numeric_columns:
            work[column] = np.nan
        else:
            work[column] = "__MISSING__"

    typed, _, _, _ = coerce_feature_types(
        work[feature_columns],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_threshold=numeric_threshold,
    )
    labels = pd.to_numeric(live_df["true_label"], errors="coerce").astype("Int64")
    labels = labels.fillna(0).astype(int)
    return typed[feature_columns], pd.Series(labels.to_numpy(), name="true_label")


def contiguous_split(n_rows: int, reference_fraction: float, drift_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reference_fraction <= 0 or drift_fraction <= 0:
        raise ValueError("reference_fraction and drift_fraction must be positive.")
    if reference_fraction + drift_fraction >= 1.0:
        raise ValueError("reference_fraction + drift_fraction must be less than 1.0.")

    ref_end = int(round(n_rows * reference_fraction))
    drift_end = int(round(n_rows * (reference_fraction + drift_fraction)))
    ref_idx = np.arange(0, ref_end)
    drift_idx = np.arange(ref_end, drift_end)
    eval_idx = np.arange(drift_end, n_rows)
    if len(ref_idx) == 0 or len(drift_idx) == 0 or len(eval_idx) == 0:
        raise ValueError("Split produced an empty batch. Adjust the split fractions.")
    return ref_idx, drift_idx, eval_idx


def prepare_feature_frame(
    df: pd.DataFrame,
    training_meta: dict[str, object],
    *,
    numeric_threshold: float,
) -> pd.DataFrame:
    feature_columns = list(training_meta["feature_columns"])
    numeric_columns = list(training_meta.get("numeric_columns", []))
    categorical_columns = list(training_meta.get("categorical_columns", []))

    from edge_iiot_experiment import coerce_feature_types, normalize_columns

    df = normalize_columns(df)
    work = pd.DataFrame(index=df.index)
    for column in feature_columns:
        if column in df.columns:
            work[column] = df[column]
        elif column in numeric_columns:
            work[column] = np.nan
        else:
            work[column] = "__MISSING__"

    typed, _, _, _ = coerce_feature_types(
        work[feature_columns],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_threshold=numeric_threshold,
    )
    return typed[feature_columns]


def evaluate_bundle(bundle: dict[str, object], X_eval: pd.DataFrame, y_eval: pd.Series, *, threshold: float) -> dict[str, object]:
    transformed = bundle["preprocessor"].transform(X_eval)
    feature_names = get_transformed_feature_names(bundle["preprocessor"])
    validate_runtime_contract(
        bundle=bundle,
        model_input=X_eval,
        transformed=transformed,
        transformed_feature_names=feature_names,
        stage="retrain_evaluate_bundle",
    )
    proba = bundle["model"].predict_proba(transformed)[:, 1]
    evaluation = evaluation_from_predictions(y_eval, proba, threshold=threshold)
    metrics = evaluation["metrics"]
    return {
        "pred_proba": proba,
        "pred_label": evaluation["pred_label"],
        "metrics": metrics,
        "confusion_matrix": evaluation["confusion_matrix"],
        "classification_report": evaluation["classification_report"],
    }


def describe_trigger(summary: dict[str, object]) -> tuple[bool, str]:
    severity = str(summary.get("severity", "low"))
    severity_stats = summary.get("severity_stats", {}) or {}
    moderate_feature_count = int(severity_stats.get("features_psi_ge_0_10", 0))
    if severity == "high":
        return True, f"severity high (max PSI {float(severity_stats.get('max_psi', 0.0)):.4f} >= {DEFAULT_HIGH_THRESHOLD:.2f})"
    if severity == "moderate" and moderate_feature_count >= DEFAULT_MODERATE_FEATURE_COUNT:
        return True, (
            f"severity moderate and {moderate_feature_count} features exceed PSI {DEFAULT_MODERATE_THRESHOLD:.2f}"
        )
    return False, f"severity {severity} did not cross the retraining rule"


def compare_metrics(original: dict[str, float], retrained: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "metric": "roc_auc",
            "original": float(original["roc_auc"]),
            "retrained": float(retrained["roc_auc"]),
            "delta": float(retrained["roc_auc"] - original["roc_auc"]),
            "direction": "higher_is_better",
        },
        {
            "metric": "pr_auc",
            "original": float(original["pr_auc"]),
            "retrained": float(retrained["pr_auc"]),
            "delta": float(retrained["pr_auc"] - original["pr_auc"]),
            "direction": "higher_is_better",
        },
        {
            "metric": "attack_precision",
            "original": float(original["precision"]),
            "retrained": float(retrained["precision"]),
            "delta": float(retrained["precision"] - original["precision"]),
            "direction": "higher_is_better",
        },
        {
            "metric": "attack_recall",
            "original": float(original["recall"]),
            "retrained": float(retrained["recall"]),
            "delta": float(retrained["recall"] - original["recall"]),
            "direction": "higher_is_better",
        },
        {
            "metric": "normal_recall",
            "original": float(original["normal_recall"]),
            "retrained": float(retrained["normal_recall"]),
            "delta": float(retrained["normal_recall"] - original["normal_recall"]),
            "direction": "higher_is_better",
        },
        {
            "metric": "macro_recall",
            "original": float(original["macro_recall"]),
            "retrained": float(retrained["macro_recall"]),
            "delta": float(retrained["macro_recall"] - original["macro_recall"]),
            "direction": "higher_is_better",
        },
        {
            "metric": "attack_fnr",
            "original": float(original["fnr"]),
            "retrained": float(retrained["fnr"]),
            "delta": float(retrained["fnr"] - original["fnr"]),
            "direction": "lower_is_better",
        },
    ]
    return pd.DataFrame(rows)


def candidate_passes_retrain_gate(
    retrained: dict[str, float],
    *,
    min_pr_auc: float = DEFAULT_MIN_RETRAIN_PR_AUC,
    min_attack_recall: float = DEFAULT_MIN_RETRAIN_ATTACK_RECALL,
    min_normal_recall: float = DEFAULT_MIN_RETRAIN_NORMAL_RECALL,
) -> tuple[bool, list[str]]:
    failures = []
    if float(retrained["pr_auc"]) < min_pr_auc:
        failures.append(f"pr_auc<{min_pr_auc:.2f}")
    if float(retrained["recall"]) < min_attack_recall:
        failures.append(f"attack_recall<{min_attack_recall:.2f}")
    if float(retrained["normal_recall"]) < min_normal_recall:
        failures.append(f"normal_recall<{min_normal_recall:.2f}")
    return not failures, failures


def assess_change(comparison: pd.DataFrame, retrained: dict[str, float] | None = None) -> str:
    if retrained is not None:
        gate_passed, _ = candidate_passes_retrain_gate(retrained)
        if not gate_passed:
            return "rejected_gate"
    pr_delta = float(comparison.loc[comparison["metric"] == "pr_auc", "delta"].iloc[0])
    recall_delta = float(comparison.loc[comparison["metric"] == "attack_recall", "delta"].iloc[0])
    fnr_delta = float(comparison.loc[comparison["metric"] == "attack_fnr", "delta"].iloc[0])
    normal_recall_delta = float(comparison.loc[comparison["metric"] == "normal_recall", "delta"].iloc[0])

    if recall_delta > 0.005 and fnr_delta < -0.005 and pr_delta >= -0.001 and normal_recall_delta >= -0.02:
        return "helped"
    if recall_delta < -0.005 or fnr_delta > 0.005 or pr_delta < -0.001 or normal_recall_delta < -0.02:
        return "hurt"
    return "little_difference"


def save_retrained_bundle(bundle: dict[str, object], model_path: Path, *, pointer_path: Path | None = None) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    attach_model_feature_names(bundle)
    joblib.dump(bundle, model_path)

    meta_path = model_path.with_suffix(".metadata.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            json_safe(
                {
                    "threshold": bundle.get("threshold"),
                    "threshold_strategy": bundle.get("threshold_strategy"),
                    "threshold_meta": bundle.get("threshold_meta", {}),
                    "fixed_threshold": bundle.get("fixed_threshold"),
                    "training_meta": bundle.get("training_meta", {}),
                    "retraining_meta": bundle.get("retraining_meta", {}),
                    "evaluation_metrics": bundle.get("evaluation_metrics", {}),
                    "runtime": bundle.get("runtime", {}),
                }
            ),
            fh,
            indent=2,
        )
    if pointer_path is not None:
        pointer = load_active_model_pointer(pointer_path) or {}
        save_active_model_pointer(
            pointer_path,
            binary_model_path=model_path,
            attack_model_path=pointer.get("attack_model_path"),
            feature_contract_path=pointer.get("feature_contract_path"),
            threshold_path=meta_path,
            multiclass_threshold_path=pointer.get("multiclass_threshold_path"),
            version=str(bundle.get("model_version") or bundle.get("retraining_meta", {}).get("model_version") or "retrained"),
            extra={
                "kind": "active_model_pointer",
                "updated_by": "adaptive_retrain",
                "retraining_assessment": bundle.get("retraining_meta", {}).get("assessment"),
            },
        )


def write_run_summary(
    *,
    output_path: Path,
    trigger: dict[str, object],
    comparison: pd.DataFrame,
    assessment: str,
    retrained_bundle_path: Path | None,
) -> None:
    lines = [
        "# Edge-IIoT Adaptive Retraining Summary",
        "",
        f"- Retraining triggered: {trigger['triggered']}",
        f"- Trigger reason: {trigger['trigger_reason']}",
        f"- Trigger severity: {trigger['severity']}",
        f"- Reference rows: {trigger['reference_rows']}",
        f"- Drift rows: {trigger['drift_rows']}",
        f"- Evaluation rows: {trigger['evaluation_rows']}",
        f"- Split strategy: {trigger['split_strategy']}",
        "",
        "## Before / After Metrics",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"- {row['metric']}: original={row['original']:.6f}, retrained={row['retrained']:.6f}, delta={row['delta']:.6f}"
        )
    lines.extend(
        [
            "",
            f"## Assessment",
            f"- Overall retraining assessment: {assessment}",
        ]
    )
    if retrained_bundle_path is not None:
        lines.append(f"- Retrained bundle: {retrained_bundle_path}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def train_command(args: argparse.Namespace) -> None:
    try:
        started = time.perf_counter()
        classifier_bundle_path = Path(args.classifier_bundle)
        original_bundle = load_bundle(classifier_bundle_path)
        threshold = float(args.threshold if args.threshold is not None else original_bundle["threshold"])
        live_df = pd.DataFrame()
        live_source_used = False
        if args.mongo_uri:
            try:
                live_df = load_recent_labeled_live_rows(args.mongo_uri, args.db_name, limit=args.recent_live_rows)
            except Exception:
                live_df = pd.DataFrame()
        if not live_df.empty and live_df["true_label"].nunique() >= 2:
            X, y = prepare_live_retraining_frame(
                live_df,
                original_bundle["training_meta"],
                numeric_threshold=args.numeric_threshold,
            )
            live_source_used = True
            split_label = "live_recent_rows"
        else:
            X, y, _ = prepare_training_frame(
                args.edge_csv,
                keep_identity_payload=args.keep_identity_payload,
                numeric_threshold=args.numeric_threshold,
                sample_rows=args.sample_rows,
                drop_duplicates=not args.keep_duplicates,
            )
            split_label = "dataset"

        ref_idx, drift_idx, eval_idx = contiguous_split(len(X), args.reference_fraction, args.drift_fraction)
        if not np.isclose(args.reference_fraction + args.drift_fraction + args.eval_fraction, 1.0, atol=0.01):
            print("Warning: split fractions do not sum exactly to 1.0; using contiguous index boundaries.")

        X_ref = X.iloc[ref_idx].reset_index(drop=True)
        y_ref = y.iloc[ref_idx].reset_index(drop=True)
        X_drift = X.iloc[drift_idx].reset_index(drop=True)
        y_drift = y.iloc[drift_idx].reset_index(drop=True)
        X_eval = X.iloc[eval_idx].reset_index(drop=True)
        y_eval = y.iloc[eval_idx].reset_index(drop=True)

        drift_feature_scores, drift_summary, _ = compare_batches(
            X_ref,
            X_drift,
            bundle=original_bundle,
            mode="live" if live_source_used else "dataset",
            reference_name="live_reference_batch" if live_source_used else "historical_reference_batch",
            target_name="live_drift_retraining_batch" if live_source_used else "drift_retraining_batch",
            n_bins=10,
            min_category_count=args.min_category_count,
            numeric_threshold=args.numeric_threshold,
        )
        triggered, trigger_reason = describe_trigger(drift_summary)

        trigger = {
            "triggered": bool(triggered),
            "trigger_reason": trigger_reason,
            "severity": drift_summary["severity"],
            "drift_rule": drift_summary["drift_rule"],
            "severity_stats": drift_summary["severity_stats"],
            "reference_rows": int(len(X_ref)),
            "drift_rows": int(len(X_drift)),
            "evaluation_rows": int(len(X_eval)),
            "split_strategy": "contiguous_60_20_20",
            "retraining_source": split_label,
            "top_features": drift_summary["top_features"][:10],
            "dataset": str(args.edge_csv),
        }

        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        trigger_path = report_dir / "edge_iiot_retrain_trigger.json"
        with trigger_path.open("w", encoding="utf-8") as fh:
            json.dump(json_safe(trigger), fh, indent=2)

        original_eval = evaluate_bundle(original_bundle, X_eval, y_eval, threshold=threshold)
        original_metrics = original_eval["metrics"]

        retrained_bundle_path: Path | None = None
        retrained_eval = None
        retrained_metrics = None
        retrained_bundle = None
        if triggered:
            training_X = pd.concat([X_ref, X_drift], ignore_index=True)
            training_y = pd.concat([y_ref, y_drift], ignore_index=True)
            training_meta = original_bundle["training_meta"]
            preprocessor = make_preprocessor(
                training_meta["numeric_columns"],
                training_meta["categorical_columns"],
                min_category_count=args.min_category_count,
            )
            X_train_tx = preprocessor.fit_transform(training_X)
            retrained_model = train_xgb_with_balance(X_train_tx, training_y, balanced_training=False)
            try:
                retrained_model.feature_names_ = get_transformed_feature_names(preprocessor)
            except Exception:
                pass
            retrained_eval = evaluate_bundle(
                {
                    "model": retrained_model,
                    "preprocessor": preprocessor,
                    "training_meta": training_meta,
                },
                X_eval,
                y_eval,
                threshold=threshold,
            )
            retrained_metrics = retrained_eval["metrics"]
            model_version = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            retrained_bundle = {
                "model": retrained_model,
                "preprocessor": preprocessor,
                "model_version": model_version,
                "threshold": threshold,
                "threshold_strategy": original_bundle.get("threshold_strategy", "fixed"),
                "fixed_threshold": threshold,
                "threshold_meta": original_bundle.get("threshold_meta", {}),
                "training_meta": {
                    **training_meta,
                    "retraining_reference_rows": int(len(X_ref)),
                    "retraining_drift_rows": int(len(X_drift)),
                    "retraining_eval_rows": int(len(X_eval)),
                    "retraining_split_strategy": "contiguous_60_20_20",
                    "drift_summary": drift_summary,
                },
                "retraining_meta": {
                    "trigger": trigger,
                    "assessment_basis": "evaluation_target_last_20_percent",
                    "reference_rows": int(len(X_ref)),
                    "drift_rows": int(len(X_drift)),
                    "evaluation_rows": int(len(X_eval)),
                    "source": split_label,
                    "model_version": model_version,
                },
                "evaluation_metrics": retrained_metrics,
                "evaluation_report": retrained_eval["classification_report"],
                "runtime": {
                    "train_seconds": float(time.perf_counter() - started),
                },
            }
            retrained_bundle_path = Path(args.model_out)
            save_retrained_bundle(retrained_bundle, retrained_bundle_path)

        comparison = compare_metrics(original_metrics, retrained_metrics if retrained_metrics is not None else original_metrics)
        gate_passed, gate_failures = (
            candidate_passes_retrain_gate(retrained_metrics) if retrained_metrics is not None else (False, ["not_retrained"])
        )
        assessment = assess_change(comparison, retrained_metrics) if triggered and retrained_metrics is not None else "not_retrained"

        pointer_updated = False
        if retrained_bundle_path is not None and gate_passed and assessment not in {"hurt", "rejected_gate"}:
            pointer = load_active_model_pointer(args.model_pointer_path) or {}
            save_active_model_pointer(
                args.model_pointer_path,
                binary_model_path=retrained_bundle_path,
                attack_model_path=pointer.get("attack_model_path"),
                feature_contract_path=pointer.get("feature_contract_path"),
                threshold_path=retrained_bundle_path.with_suffix(".metadata.json"),
                multiclass_threshold_path=pointer.get("multiclass_threshold_path"),
                version=str(
                    retrained_bundle.get("model_version")
                    or retrained_bundle.get("retraining_meta", {}).get("model_version")
                    or "retrained"
                ),
                extra={
                    "kind": "active_model_pointer",
                    "updated_by": "adaptive_retrain",
                    "retraining_assessment": assessment,
                },
            )
            pointer_updated = True

        comparison_path = report_dir / "edge_iiot_retrain_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        comparison_json_path = report_dir / "edge_iiot_retrain_comparison.json"
        with comparison_json_path.open("w", encoding="utf-8") as fh:
            json.dump(
                json_safe(
                    {
                        "triggered": bool(triggered),
                        "trigger_reason": trigger_reason,
                        "severity": drift_summary["severity"],
                        "assessment": assessment,
                        "candidate_gate_passed": gate_passed,
                        "candidate_gate_failures": gate_failures,
                        "candidate_gate": {
                            "min_pr_auc": DEFAULT_MIN_RETRAIN_PR_AUC,
                            "min_attack_recall": DEFAULT_MIN_RETRAIN_ATTACK_RECALL,
                            "min_normal_recall": DEFAULT_MIN_RETRAIN_NORMAL_RECALL,
                        },
                        "pointer_updated": pointer_updated,
                        "original_metrics": original_metrics,
                        "retrained_metrics": retrained_metrics,
                        "comparison": comparison.to_dict(orient="records"),
                        "retraining_source": split_label,
                    }
                ),
                fh,
                indent=2,
            )

        write_run_summary(
            output_path=report_dir / "edge_iiot_retrain_run_summary.md",
            trigger=trigger,
            comparison=comparison,
            assessment=assessment,
            retrained_bundle_path=retrained_bundle_path,
        )

        if args.mongo_uri:
            try:
                db = connect_database(args.mongo_uri, args.db_name)
                ensure_doc = {
                    "kind": "offline_retrain_result",
                    "source": "adaptive_retrain",
                    "created_at": datetime.now(timezone.utc),
                    "triggered": bool(triggered),
                    "trigger_reason": trigger_reason,
                    "severity": drift_summary["severity"],
                    "assessment": assessment,
                    "candidate_gate_passed": gate_passed,
                    "candidate_gate_failures": gate_failures,
                    "pointer_updated": pointer_updated,
                    "retraining_source": split_label,
                    "original_metrics": original_metrics,
                    "retrained_metrics": retrained_metrics,
                    "comparison": comparison.to_dict(orient="records"),
                    "model_path": str(retrained_bundle_path) if retrained_bundle_path else None,
                    "pointer_path": str(args.model_pointer_path),
                }
                insert_documents(db[RETRAIN_EVENTS_COLLECTION], [ensure_doc])
            except Exception:
                pass

        print("Adaptive retraining complete.")
        print(f"Triggered         : {triggered}")
        print(f"Trigger reason    : {trigger_reason}")
        print(f"Severity          : {drift_summary['severity']}")
        print("Original metrics:")
        print(pd.Series(original_metrics)[["roc_auc", "pr_auc", "recall", "normal_recall", "macro_recall", "fnr"]].to_string())
        if retrained_metrics is not None:
            print("Retrained metrics:")
            print(pd.Series(retrained_metrics)[["roc_auc", "pr_auc", "recall", "normal_recall", "macro_recall", "fnr"]].to_string())
            print(f"Candidate gate    : {gate_passed} {gate_failures}")
        print(f"Assessment        : {assessment}")
        print(f"Trigger report    : {trigger_path}")
        print(f"Comparison report : {comparison_json_path}")
        print(f"Comparison CSV    : {comparison_path}")
        if retrained_bundle_path is not None:
            print(f"Retrained model   : {retrained_bundle_path}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline adaptive retraining workflow for Edge-IIoT.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run conditional drift-triggered retraining.")
    train_parser.add_argument("--edge_csv", default=DEFAULT_EDGE_CSV)
    train_parser.add_argument("--classifier_bundle", default=str(DEFAULT_CLASSIFIER_BUNDLE))
    train_parser.add_argument("--model_out", default=str(DEFAULT_RETRAINED_BUNDLE))
    train_parser.add_argument("--report_dir", default=str(DEFAULT_REPORT_DIR))
    train_parser.add_argument("--reference_fraction", type=float, default=DEFAULT_REFERENCE_FRACTION)
    train_parser.add_argument("--drift_fraction", type=float, default=DEFAULT_DRIFT_FRACTION)
    train_parser.add_argument("--eval_fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    train_parser.add_argument("--numeric_threshold", type=float, default=0.95)
    train_parser.add_argument("--min_category_count", type=int, default=20)
    train_parser.add_argument("--sample_rows", type=int, default=None)
    train_parser.add_argument("--keep_duplicates", action="store_true")
    train_parser.add_argument("--keep_identity_payload", action="store_true")
    train_parser.add_argument("--random_state", type=int, default=42)
    train_parser.add_argument("--threshold", type=float, default=None)
    train_parser.add_argument("--mongo_uri", default=None)
    train_parser.add_argument("--db_name", default="edge_iiot_paper")
    train_parser.add_argument("--recent_live_rows", type=int, default=DEFAULT_RECENT_LIVE_ROWS)
    train_parser.add_argument("--model_pointer_path", default=str(DEFAULT_MODEL_POINTER_PATH))
    train_parser.set_defaults(func=train_command)

    args = parser.parse_args()
    if args.command != "train":
        raise SystemExit("Only the train command is supported for offline adaptive retraining.")
    args.func(args)


if __name__ == "__main__":
    main()
