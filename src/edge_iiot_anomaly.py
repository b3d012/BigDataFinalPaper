from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse as scipy_sparse
from sklearn.ensemble import IsolationForest

from edge_iiot_experiment import (
    DEFAULT_EDGE_CSV,
    evaluation_from_predictions,
    get_transformed_feature_names,
    make_preprocessor,
    prepare_training_frame,
    coerce_feature_types,
    normalize_columns,
)
from edge_iiot_runtime import attach_model_feature_names, validate_runtime_contract


DEFAULT_CLASSIFIER_BUNDLE = Path("models/edge_iiot_xgb_model.joblib")
DEFAULT_ANOMALY_BUNDLE = Path("models/edge_iiot_isolation_forest.joblib")
DEFAULT_REPORT_DIR = Path("output/reports")
DEFAULT_DEMO_INPUT = Path("output/demo/edge_iiot_demo_predictions.csv")
DEFAULT_DEMO_OUTPUT_DIR = Path("output/demo")
DEFAULT_DEMO_PREDICTIONS = DEFAULT_DEMO_OUTPUT_DIR / "edge_iiot_demo_anomaly_predictions.csv"
DEFAULT_DEMO_SUMMARY = DEFAULT_DEMO_OUTPUT_DIR / "edge_iiot_demo_anomaly_summary.csv"
DEFAULT_DEMO_COMPARISON = DEFAULT_DEMO_OUTPUT_DIR / "edge_iiot_demo_anomaly_vs_classifier.csv"
DEFAULT_RUN_SUMMARY = DEFAULT_REPORT_DIR / "edge_iiot_anomaly_run_summary.md"

DEFAULT_CONTAMINATION = 0.05
DEFAULT_BENIGN_SCORE_QUANTILE = 0.05
DEFAULT_FILE_MAX_SCORE_THRESHOLD = None
DEFAULT_FILE_RATIO_THRESHOLD = 0.4
DEFAULT_MIN_FILE_RECORDS = 0


def load_bundle(model_path: Path) -> dict[str, object]:
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier bundle not found: {model_path}")
    bundle = joblib.load(model_path)
    required = {"model", "preprocessor", "threshold", "training_meta"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Classifier bundle is missing required keys: {sorted(missing)}")
    attach_model_feature_names(bundle)
    return bundle


def load_classifier_holdout_predictions(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    return normalize_columns(df)


def prepare_model_input(df: pd.DataFrame, training_meta: dict[str, object]) -> pd.DataFrame:
    df = normalize_columns(df)
    feature_columns = list(training_meta["feature_columns"])
    numeric_columns = list(training_meta.get("numeric_columns", []))
    categorical_columns = list(training_meta.get("categorical_columns", []))

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
    )
    return typed[feature_columns]


def make_anomaly_scores(model: IsolationForest, matrix, *, dense_used: bool) -> np.ndarray:
    if dense_used and scipy_sparse.issparse(matrix):
        matrix = matrix.toarray()
    scores = model.decision_function(matrix)
    return -np.asarray(scores, dtype=float)


def fit_isolation_forest(
    X_train_tx,
    y_train: pd.Series,
    *,
    contamination: float,
    random_state: int,
    n_estimators: int,
) -> tuple[IsolationForest, bool]:
    benign_mask = (y_train == 0).to_numpy()
    X_train_benign = X_train_tx[benign_mask]

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    dense_used = False
    try:
        model.fit(X_train_benign)
    except Exception as exc:
        if not scipy_sparse.issparse(X_train_benign):
            raise
        dense_used = True
        model.fit(X_train_benign.toarray())
        print(
            "IsolationForest fell back to dense input because the sparse matrix path raised "
            f"{type(exc).__name__}. The fitted model is unchanged."
        )
    return model, dense_used


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {}
    return {
        "count": int(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "q05": float(np.quantile(scores, 0.05)),
        "q25": float(np.quantile(scores, 0.25)),
        "median": float(np.quantile(scores, 0.50)),
        "q75": float(np.quantile(scores, 0.75)),
        "q95": float(np.quantile(scores, 0.95)),
        "max": float(np.max(scores)),
    }


def build_holdout_split(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    return train_idx, test_idx


def evaluate_anomaly_predictions(
    y_true: pd.Series,
    anomaly_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    return evaluation_from_predictions(y_true, anomaly_score, threshold=threshold)


def build_holdout_comparison(
    holdout_predictions: pd.DataFrame,
    classifier_holdout_predictions: pd.DataFrame | None,
) -> pd.DataFrame:
    comparison = holdout_predictions.copy()
    if classifier_holdout_predictions is None:
        comparison["classifier_pred_label"] = np.nan
        comparison["classifier_pred_proba_attack"] = np.nan
        comparison["agreement"] = np.nan
        return comparison

    if len(classifier_holdout_predictions) != len(comparison):
        raise ValueError(
            "Classifier holdout predictions do not match the anomaly holdout split length. "
            f"Classifier rows={len(classifier_holdout_predictions)} anomaly rows={len(comparison)}"
        )

    comparison["classifier_pred_label"] = classifier_holdout_predictions["pred_label"].to_numpy()
    comparison["classifier_pred_proba_attack"] = classifier_holdout_predictions["pred_proba_attack"].to_numpy()
    comparison["agreement"] = (comparison["classifier_pred_label"] == comparison["anomaly_pred_label"]).astype(int)
    comparison["classifier_correct"] = (comparison["classifier_pred_label"] == comparison["true_label"]).astype(int)
    comparison["anomaly_correct"] = (comparison["anomaly_pred_label"] == comparison["true_label"]).astype(int)
    return comparison


def build_demo_summary(
    predictions: pd.DataFrame,
    *,
    file_max_score_threshold: float,
    file_ratio_threshold: float,
    min_records: int,
) -> pd.DataFrame:
    if "source_file" not in predictions.columns:
        predictions = predictions.copy()
        predictions["source_file"] = "unknown"

    agg_kwargs: dict[str, tuple[str, object]] = {
        "records": ("anomaly_pred_label", "size"),
        "anomaly_records": ("anomaly_pred_label", "sum"),
        "anomaly_record_ratio": ("anomaly_pred_label", "mean"),
        "mean_anomaly_score": ("anomaly_score", "mean"),
        "median_anomaly_score": ("anomaly_score", "median"),
        "p95_anomaly_score": ("anomaly_score", lambda s: float(s.quantile(0.95))),
        "max_anomaly_score": ("anomaly_score", "max"),
    }
    if "classifier_pred_label" in predictions.columns:
        agg_kwargs["classifier_attack_records"] = ("classifier_pred_label", "sum")
        agg_kwargs["classifier_attack_ratio"] = ("classifier_pred_label", "mean")
    if "agreement" in predictions.columns:
        agg_kwargs["agreement_rate"] = ("agreement", "mean")

    summary = predictions.groupby("source_file", sort=True).agg(**agg_kwargs).reset_index()
    summary["file_pass_max_score_rule"] = summary["max_anomaly_score"] >= file_max_score_threshold
    summary["file_pass_ratio_rule"] = summary["anomaly_record_ratio"] >= file_ratio_threshold
    summary["file_pred_label"] = (
        (summary["records"] >= min_records)
        & summary["file_pass_max_score_rule"]
        & summary["file_pass_ratio_rule"]
    ).astype(int)
    return summary.sort_values(["anomaly_record_ratio", "source_file"], ascending=[True, True], kind="mergesort")


def prepare_demo_predictions(
    df: pd.DataFrame,
    bundle: dict[str, object],
    *,
    dense_used: bool,
    threshold: float,
) -> pd.DataFrame:
    training_meta = bundle["training_meta"]
    model_input = prepare_model_input(df, training_meta)
    transformed = bundle["preprocessor"].transform(model_input)
    validate_runtime_contract(
        bundle=bundle,
        model_input=model_input,
        transformed=transformed,
        transformed_feature_names=get_transformed_feature_names(bundle["preprocessor"]),
        stage="anomaly_demo_score",
    )
    scores = make_anomaly_scores(bundle["model"], transformed, dense_used=dense_used)
    pred_label = (scores >= threshold).astype(int)

    out = normalize_columns(df).copy()
    out["anomaly_score"] = scores
    out["anomaly_pred_label"] = pred_label
    out["anomaly_pred_name"] = out["anomaly_pred_label"].map({0: "Normal", 1: "Anomaly"})
    if "pred_label" in out.columns:
        out["classifier_pred_label"] = pd.to_numeric(out["pred_label"], errors="coerce")
        out["agreement"] = (out["classifier_pred_label"] == out["anomaly_pred_label"]).astype(int)
    return out


def fit_anomaly_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    training_meta: dict[str, object],
    *,
    contamination: float,
    benign_score_quantile: float,
    min_category_count: int,
    random_state: int,
    n_estimators: int,
    classifier_holdout_predictions: pd.DataFrame | None,
) -> dict[str, object]:
    train_idx, test_idx = build_holdout_split(X, y)
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    preprocessor = make_preprocessor(
        training_meta["numeric_columns"],
        training_meta["categorical_columns"],
        min_category_count=min_category_count,
    )
    preprocessor_fit_started = time.perf_counter()
    X_train_tx = preprocessor.fit_transform(X_train)
    X_test_tx = preprocessor.transform(X_test)
    preprocessor_fit_seconds = time.perf_counter() - preprocessor_fit_started

    anomaly_model, dense_used = fit_isolation_forest(
        X_train_tx,
        y_train,
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
    )

    train_benign_mask = (y_train == 0).to_numpy()
    train_benign_scores = make_anomaly_scores(anomaly_model, X_train_tx[train_benign_mask], dense_used=dense_used)
    test_scores = make_anomaly_scores(anomaly_model, X_test_tx, dense_used=dense_used)
    threshold = float(np.quantile(train_benign_scores, benign_score_quantile))
    pred_label = (test_scores >= threshold).astype(int)

    evaluation = evaluate_anomaly_predictions(y_test, test_scores, threshold=threshold)
    metrics = evaluation["metrics"]
    confusion = evaluation["confusion_matrix"]
    report = evaluation["classification_report"]

    holdout_predictions = pd.DataFrame(
        {
            "dataset_row_index": test_idx,
            "true_label": y_test.to_numpy(),
            "true_label_name": y_test.map({0: "Normal", 1: "Attack"}).to_numpy(),
            "anomaly_score": test_scores,
            "anomaly_pred_label": pred_label,
            "anomaly_pred_name": pd.Series(pred_label).map({0: "Normal", 1: "Anomaly"}).to_numpy(),
        }
    )
    holdout_comparison = build_holdout_comparison(holdout_predictions, classifier_holdout_predictions)

    transformed_feature_names = get_transformed_feature_names(preprocessor)
    try:
        anomaly_model.feature_names_ = list(transformed_feature_names)
    except Exception:
        pass
    runtime = {
        "rows_total": int(len(X)),
        "features_total": int(X.shape[1]),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_benign_rows": int(train_benign_mask.sum()),
        "train_attack_rows": int((~train_benign_mask).sum()),
        "preprocessor_fit_seconds": float(preprocessor_fit_seconds),
        "dense_fallback_used": bool(dense_used),
    }

    score_summary = {
        "train_benign_anomaly_score": summarize_scores(train_benign_scores),
        "holdout_anomaly_score": summarize_scores(test_scores),
    }

    return {
        "model": anomaly_model,
        "preprocessor": preprocessor,
        "threshold": threshold,
        "threshold_strategy": f"benign_score_quantile_{benign_score_quantile:.4f}",
        "contamination": float(contamination),
        "benign_score_quantile": float(benign_score_quantile),
        "training_meta": training_meta,
        "evaluation_metrics": metrics,
        "evaluation_report": report,
        "confusion_matrix": confusion,
        "holdout_predictions": holdout_predictions,
        "holdout_comparison": holdout_comparison,
        "score_summary": score_summary,
        "runtime": runtime,
        "transformed_feature_names": transformed_feature_names,
    }


def save_reports(bundle: dict[str, object], report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = report_dir / "edge_iiot_anomaly_holdout_metrics.json"
    predictions_path = report_dir / "edge_iiot_anomaly_holdout_predictions.csv"
    confusion_path = report_dir / "edge_iiot_anomaly_confusion_matrix.csv"
    comparison_path = report_dir / "edge_iiot_anomaly_classifier_comparison.csv"
    class_report_path = report_dir / "edge_iiot_anomaly_classification_report.csv"

    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "threshold": bundle["threshold"],
                "threshold_strategy": bundle["threshold_strategy"],
                "contamination": bundle["contamination"],
                "benign_score_quantile": bundle["benign_score_quantile"],
                "training_meta": bundle["training_meta"],
                "evaluation_metrics": bundle["evaluation_metrics"],
                "score_summary": bundle["score_summary"],
                "runtime": bundle["runtime"],
                "transformed_feature_names": bundle["transformed_feature_names"],
            },
            fh,
            indent=2,
        )

    bundle["holdout_predictions"].to_csv(predictions_path, index=False)
    pd.DataFrame(
        bundle["confusion_matrix"],
        index=["true_normal", "true_attack"],
        columns=["pred_normal", "pred_anomaly"],
    ).to_csv(confusion_path)
    bundle["holdout_comparison"].to_csv(comparison_path, index=False)
    pd.DataFrame(bundle["evaluation_report"]).T.to_csv(class_report_path)

    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "confusion_matrix": confusion_path,
        "comparison": comparison_path,
        "classification_report": class_report_path,
    }


def save_anomaly_bundle(bundle: dict[str, object], model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle["model"],
            "preprocessor": bundle["preprocessor"],
            "threshold": bundle["threshold"],
            "threshold_strategy": bundle["threshold_strategy"],
            "contamination": bundle["contamination"],
            "benign_score_quantile": bundle["benign_score_quantile"],
            "training_meta": bundle["training_meta"],
            "evaluation_metrics": bundle["evaluation_metrics"],
            "evaluation_report": bundle["evaluation_report"],
            "score_summary": bundle["score_summary"],
            "runtime": bundle["runtime"],
            "transformed_feature_names": bundle["transformed_feature_names"],
        },
        model_path,
    )

    meta_path = model_path.with_suffix(".metadata.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "threshold": bundle["threshold"],
                "threshold_strategy": bundle["threshold_strategy"],
                "contamination": bundle["contamination"],
                "benign_score_quantile": bundle["benign_score_quantile"],
                "training_meta": bundle["training_meta"],
                "evaluation_metrics": bundle["evaluation_metrics"],
                "score_summary": bundle["score_summary"],
                "runtime": bundle["runtime"],
                "transformed_feature_names": bundle["transformed_feature_names"],
            },
            fh,
            indent=2,
        )


def write_run_summary(
    *,
    output_path: Path,
    classifier_bundle_path: Path,
    model_path: Path,
    bundle: dict[str, object],
    demo_summary: pd.DataFrame | None,
) -> None:
    metrics = bundle["evaluation_metrics"]
    score_summary = bundle["score_summary"]

    lines = [
        "# Edge-IIoT Anomaly Layer Summary",
        "",
        f"- Classifier bundle: {classifier_bundle_path}",
        f"- Anomaly bundle: {model_path}",
        f"- Threshold strategy: {bundle['threshold_strategy']}",
        f"- Decision threshold: {bundle['threshold']:.6f}",
        f"- Contamination: {bundle['contamination']:.4f}",
        f"- Benign score quantile: {bundle['benign_score_quantile']:.4f}",
        f"- Dense fallback used: {bundle['runtime']['dense_fallback_used']}",
        "",
        "## Holdout Metrics",
        f"- ROC-AUC: {metrics['roc_auc']:.4f}",
        f"- PR-AUC: {metrics['pr_auc']:.4f}",
        f"- Attack precision: {metrics['precision']:.4f}",
        f"- Attack recall: {metrics['recall']:.4f}",
        f"- Normal recall: {metrics['normal_recall']:.4f}",
        f"- Macro recall: {metrics['macro_recall']:.4f}",
        f"- Attack FNR: {metrics['fnr']:.4f}",
        "",
        "## Score Summary",
    ]

    for label, stats in score_summary.items():
        lines.append(
            f"- {label}: count={stats.get('count', 0)} "
            f"mean={stats.get('mean', float('nan')):.6f} std={stats.get('std', float('nan')):.6f}"
        )

    if demo_summary is not None and not demo_summary.empty:
        lines.extend(
            [
                "",
                "## Demo Replay",
                f"- Files scored: {len(demo_summary)}",
                f"- Demo attack file predictions: {int(demo_summary['file_pred_label'].sum())}",
                f"- Demo benign file predictions: {int((demo_summary['file_pred_label'] == 0).sum())}",
            ]
        )
    else:
        lines.extend(["", "## Demo Replay", "- Demo scoring was skipped because no demo input was available."])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def train_command(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    classifier_bundle_path = Path(args.classifier_bundle)
    classifier_bundle = load_bundle(classifier_bundle_path)
    demo_input_path = Path(args.demo_input)
    demo_output_dir = Path(args.demo_output_dir)
    report_dir = Path(args.report_dir)
    model_out = Path(args.model_out)

    X, y, derived_meta = prepare_training_frame(
        args.edge_csv,
        keep_identity_payload=args.keep_identity_payload,
        numeric_threshold=args.numeric_threshold,
        sample_rows=args.sample_rows,
        drop_duplicates=not args.keep_duplicates,
    )
    training_meta = classifier_bundle["training_meta"]
    feature_columns = list(training_meta["feature_columns"])
    missing_features = [column for column in feature_columns if column not in X.columns]
    if missing_features:
        raise ValueError(
            "The current dataset does not contain all features from the saved classifier bundle. "
            f"Missing columns: {missing_features[:10]}"
        )
    X = X[feature_columns]
    if list(derived_meta["feature_columns"]) != feature_columns:
        print("Warning: derived feature columns differ from the saved classifier bundle contract; using the saved contract.")

    print("\nAnomaly training data:")
    print(f"Rows                   : {len(X):,}")
    print(f"Features used          : {X.shape[1]:,}")
    print(f"Numeric features       : {len(training_meta['numeric_columns']):,}")
    print(f"Categorical features   : {len(training_meta['categorical_columns']):,}")
    print(f"Classifier bundle      : {classifier_bundle_path}")
    print(f"Contamination          : {args.contamination:.4f}")
    print(f"Benign score quantile  : {args.benign_score_quantile:.4f}")

    bundle = fit_anomaly_bundle(
        X,
        y,
        training_meta,
        contamination=args.contamination,
        benign_score_quantile=args.benign_score_quantile,
        min_category_count=args.min_category_count,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        classifier_holdout_predictions=load_classifier_holdout_predictions(
            report_dir / "edge_iiot_holdout_predictions.csv"
        ),
    )
    bundle["runtime"]["total_fit_seconds"] = float(time.perf_counter() - started)

    save_anomaly_bundle(bundle, model_out)
    print(f"\nSaved anomaly bundle: {model_out}")
    print(f"Saved anomaly metadata: {model_out.with_suffix('.metadata.json')}")

    report_paths = save_reports(bundle, report_dir)
    print("Saved anomaly reports:")
    for label, path in report_paths.items():
        print(f"  {label}: {path}")

    demo_summary = None
    if demo_input_path.exists():
        demo_output_dir.mkdir(parents=True, exist_ok=True)
        demo_df = pd.read_csv(demo_input_path, low_memory=False)
        demo_df = normalize_columns(demo_df)
        file_max_score_threshold = (
            float(args.file_max_score_threshold)
            if args.file_max_score_threshold is not None
            else float(bundle["threshold"])
        )
        demo_predictions = prepare_demo_predictions(
            demo_df,
            bundle,
            dense_used=bool(bundle["runtime"]["dense_fallback_used"]),
            threshold=float(bundle["threshold"]),
        )
        demo_predictions_path = demo_output_dir / DEFAULT_DEMO_PREDICTIONS.name
        demo_summary_path = demo_output_dir / DEFAULT_DEMO_SUMMARY.name
        demo_comparison_path = demo_output_dir / DEFAULT_DEMO_COMPARISON.name
        demo_predictions.to_csv(demo_predictions_path, index=False)
        demo_summary = build_demo_summary(
            demo_predictions,
            file_max_score_threshold=file_max_score_threshold,
            file_ratio_threshold=args.file_ratio_threshold,
            min_records=args.min_file_records,
        )
        demo_summary.to_csv(demo_summary_path, index=False)
        if "agreement" in demo_predictions.columns:
            demo_comparison = demo_predictions[
                [
                    *[col for col in ("source_file", "record_index") if col in demo_predictions.columns],
                    "anomaly_score",
                    "anomaly_pred_label",
                    "anomaly_pred_name",
                    "pred_proba_attack",
                    "pred_label",
                    "agreement",
                ]
            ].copy()
            demo_comparison.to_csv(demo_comparison_path, index=False)
            print(f"Saved demo comparison: {demo_comparison_path}")
        print(f"Saved demo anomaly predictions: {demo_predictions_path}")
        print(f"Saved demo anomaly summary: {demo_summary_path}")
    else:
        print(f"Demo input not found, skipping demo scoring: {demo_input_path}")

    run_summary_path = report_dir / DEFAULT_RUN_SUMMARY.name
    write_run_summary(
        output_path=run_summary_path,
        classifier_bundle_path=classifier_bundle_path,
        model_path=model_out,
        bundle=bundle,
        demo_summary=demo_summary,
    )
    print(f"Saved run summary: {run_summary_path}")
    print("\nHoldout anomaly metrics:")
    print(f"ROC-AUC    : {bundle['evaluation_metrics']['roc_auc']:.4f}")
    print(f"PR-AUC     : {bundle['evaluation_metrics']['pr_auc']:.4f}")
    print(f"Precision  : {bundle['evaluation_metrics']['precision']:.4f}")
    print(f"Recall     : {bundle['evaluation_metrics']['recall']:.4f}")
    print(f"Normal R   : {bundle['evaluation_metrics']['normal_recall']:.4f}")
    print(f"Macro R    : {bundle['evaluation_metrics']['macro_recall']:.4f}")
    print(f"FNR        : {bundle['evaluation_metrics']['fnr']:.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]")
    print(
        pd.DataFrame(
            bundle["confusion_matrix"],
            index=["true_normal", "true_attack"],
            columns=["pred_normal", "pred_anomaly"],
        ).to_string()
    )

    comparison = bundle.get("holdout_comparison")
    if isinstance(comparison, pd.DataFrame) and "agreement" in comparison.columns:
        agreement_rate = float(pd.to_numeric(comparison["agreement"], errors="coerce").mean())
        print(f"Classifier/anomaly holdout agreement: {agreement_rate:.4f}")

    print("\nTrain command complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Edge-IIoT anomaly detector built on the trained feature contract.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the offline anomaly detector.")
    train_parser.add_argument("--edge_csv", default=DEFAULT_EDGE_CSV)
    train_parser.add_argument("--classifier_bundle", default=str(DEFAULT_CLASSIFIER_BUNDLE))
    train_parser.add_argument("--model_out", default=str(DEFAULT_ANOMALY_BUNDLE))
    train_parser.add_argument("--report_dir", default=str(DEFAULT_REPORT_DIR))
    train_parser.add_argument("--demo_input", default=str(DEFAULT_DEMO_INPUT))
    train_parser.add_argument("--demo_output_dir", default=str(DEFAULT_DEMO_OUTPUT_DIR))
    train_parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    train_parser.add_argument("--benign_score_quantile", type=float, default=DEFAULT_BENIGN_SCORE_QUANTILE)
    train_parser.add_argument("--numeric_threshold", type=float, default=0.95)
    train_parser.add_argument("--min_category_count", type=int, default=20)
    train_parser.add_argument("--sample_rows", type=int, default=None)
    train_parser.add_argument("--keep_duplicates", action="store_true")
    train_parser.add_argument("--keep_identity_payload", action="store_true")
    train_parser.add_argument("--random_state", type=int, default=42)
    train_parser.add_argument("--n_estimators", type=int, default=200)
    train_parser.add_argument("--file_max_score_threshold", type=float, default=DEFAULT_FILE_MAX_SCORE_THRESHOLD)
    train_parser.add_argument("--file_ratio_threshold", type=float, default=DEFAULT_FILE_RATIO_THRESHOLD)
    train_parser.add_argument("--min_file_records", type=int, default=DEFAULT_MIN_FILE_RECORDS)
    train_parser.set_defaults(func=train_command)

    args = parser.parse_args()
    if args.command != "train":
        raise SystemExit("Only the train command is supported in this offline anomaly layer.")

    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
