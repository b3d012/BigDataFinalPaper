from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from edge_iiot_experiment import DEFAULT_REPORT_DIR
from edge_iiot_multiclass import calibrate_multiclass_thresholds


DEFAULT_FIGURE_DIR = Path("output/figures")
DEFAULT_HOLDOUT_PREDICTIONS = DEFAULT_REPORT_DIR / "edge_iiot_holdout_predictions.csv"
DEFAULT_CV_PREDICTIONS = DEFAULT_REPORT_DIR / "edge_iiot_cv_predictions.csv"
DEFAULT_HOLDOUT_GRID = DEFAULT_REPORT_DIR / "edge_iiot_holdout_threshold_grid.csv"
DEFAULT_HOLDOUT_RECOMMENDATIONS = DEFAULT_REPORT_DIR / "edge_iiot_holdout_threshold_recommendations.json"
DEFAULT_CV_GRID = DEFAULT_REPORT_DIR / "edge_iiot_cv_threshold_grid.csv"
DEFAULT_CV_RECOMMENDATIONS = DEFAULT_REPORT_DIR / "edge_iiot_cv_threshold_recommendations.json"
DEFAULT_SUMMARY_MD = DEFAULT_REPORT_DIR / "edge_iiot_threshold_calibration_summary.md"


def build_threshold_grid(
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> np.ndarray:
    if threshold_min < 0 or threshold_max > 1 or threshold_min >= threshold_max:
        raise ValueError("Threshold range must satisfy 0 <= min < max <= 1.")
    if threshold_step <= 0:
        raise ValueError("Threshold step must be positive.")
    grid = np.round(np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step), 6)
    grid = grid[(grid >= threshold_min) & (grid <= threshold_max)]
    if len(grid) == 0:
        raise ValueError("Threshold grid is empty.")
    return grid


def load_prediction_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = {"true_label", "pred_proba_attack"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file {path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["true_label"] = pd.to_numeric(df["true_label"], errors="raise").astype(int)
    df["pred_proba_attack"] = pd.to_numeric(df["pred_proba_attack"], errors="raise").astype(float)
    return df


def compute_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    for threshold in thresholds:
        pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        normal_recall = float(tn / (tn + fp)) if tn + fp else 0.0
        macro_recall = float((normal_recall + recall) / 2.0)
        fnr = float(fn / (fn + tp)) if fn + tp else 0.0
        f1 = float((2 * precision * recall) / (precision + recall + 1e-12))
        f2 = float((5 * precision * recall) / (4 * precision + recall + 1e-12))
        rows.append(
            {
                "threshold": float(threshold),
                "attack_precision": precision,
                "attack_recall": recall,
                "normal_recall": normal_recall,
                "macro_recall": macro_recall,
                "attack_fnr": fnr,
                "attack_f1": f1,
                "attack_f2": f2,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "roc_auc_context": roc_auc,
                "pr_auc_context": pr_auc,
            }
        )
    return pd.DataFrame(rows)


def row_to_metrics(row: pd.Series) -> dict[str, float]:
    return {
        "threshold": float(row["threshold"]),
        "attack_precision": float(row["attack_precision"]),
        "attack_recall": float(row["attack_recall"]),
        "normal_recall": float(row["normal_recall"]),
        "macro_recall": float(row["macro_recall"]),
        "attack_fnr": float(row["attack_fnr"]),
        "attack_f1": float(row["attack_f1"]),
        "attack_f2": float(row["attack_f2"]),
        "tn": int(row["tn"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "tp": int(row["tp"]),
    }


def select_recommendations(grid: pd.DataFrame, *, min_precision: float, default_threshold: float) -> dict[str, object]:
    default_idx = (grid["threshold"] - default_threshold).abs().idxmin()
    default_row = grid.loc[default_idx]

    feasible = grid.loc[grid["attack_precision"] >= min_precision].copy()
    feasible_met = not feasible.empty
    if feasible.empty:
        feasible = grid.copy()

    max_f1_row = grid.loc[grid["attack_f1"].idxmax()]
    max_f2_row = grid.loc[grid["attack_f2"].idxmax()]
    best_recall_row = feasible.sort_values(
        ["attack_recall", "attack_precision", "threshold"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    lowest_fnr_row = feasible.sort_values(
        ["attack_fnr", "threshold", "attack_precision"],
        ascending=[True, True, False],
        kind="mergesort",
    ).iloc[0]

    def pack(row: pd.Series, *, note: str, precision_constraint_met: bool | None = None) -> dict[str, object]:
        metrics = row_to_metrics(row)
        metrics["delta_from_default"] = {
            "attack_precision": float(row["attack_precision"] - default_row["attack_precision"]),
            "attack_recall": float(row["attack_recall"] - default_row["attack_recall"]),
            "normal_recall": float(row["normal_recall"] - default_row["normal_recall"]),
            "macro_recall": float(row["macro_recall"] - default_row["macro_recall"]),
            "attack_fnr": float(row["attack_fnr"] - default_row["attack_fnr"]),
            "attack_f1": float(row["attack_f1"] - default_row["attack_f1"]),
            "attack_f2": float(row["attack_f2"] - default_row["attack_f2"]),
        }
        metrics["note"] = note
        if precision_constraint_met is not None:
            metrics["precision_constraint_met"] = bool(precision_constraint_met)
        return metrics

    return {
        "default_threshold": row_to_metrics(default_row),
        "min_precision_constraint": float(min_precision),
        "precision_constraint_met": bool(feasible_met),
        "recommended": {
            "max_f1": pack(max_f1_row, note="Threshold with the highest F1 on the evaluated probabilities."),
            "max_f2": pack(max_f2_row, note="Threshold with the highest F2 to emphasize recall."),
            "highest_recall_under_min_precision": pack(
                best_recall_row,
                note="Best recall subject to the minimum precision constraint.",
                precision_constraint_met=feasible_met,
            ),
            "lowest_fnr_under_min_precision": pack(
                lowest_fnr_row,
                note="Lowest FNR subject to the minimum precision constraint.",
                precision_constraint_met=feasible_met,
            ),
        },
        "default_row_metrics": row_to_metrics(default_row),
    }


def write_recommendations(
    *,
    grid: pd.DataFrame,
    recommendations: dict[str, object],
    output_path: Path,
    source_name: str,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> None:
    payload = {
        "source_name": source_name,
        "threshold_grid": {
            "min": float(threshold_min),
            "max": float(threshold_max),
            "step": float(threshold_step),
            "count": int(len(grid)),
        },
        **recommendations,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_threshold_figures(
    *,
    grid: pd.DataFrame,
    recommendations: dict[str, object],
    figure_prefix: str,
    figure_dir: Path,
) -> tuple[Path, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(
            "matplotlib is required for threshold figures. Install it with `pip install matplotlib`."
        ) from exc

    figure_dir.mkdir(parents=True, exist_ok=True)
    recommended = recommendations["recommended"]
    default_thr = float(recommendations["default_row_metrics"]["threshold"])

    precision_recall_path = figure_dir / f"{figure_prefix}_precision_recall_vs_threshold.png"
    fnr_path = figure_dir / f"{figure_prefix}_fnr_vs_threshold.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grid["threshold"], grid["attack_precision"], label="Attack precision", color="#1f77b4")
    ax.plot(grid["threshold"], grid["attack_recall"], label="Attack recall", color="#ff7f0e")
    ax.plot(grid["threshold"], grid["attack_f1"], label="F1", color="#2ca02c", alpha=0.8)
    ax.plot(grid["threshold"], grid["attack_f2"], label="F2", color="#9467bd", alpha=0.8)
    ax.axvline(default_thr, color="#444444", linestyle="--", linewidth=1.5, label="Default threshold")
    ax.axvline(
        recommended["highest_recall_under_min_precision"]["threshold"],
        color="#d62728",
        linestyle=":",
        linewidth=2,
        label="Recall under precision constraint",
    )
    ax.axvline(
        recommended["max_f2"]["threshold"],
        color="#17becf",
        linestyle="--",
        linewidth=1.5,
        label="Max F2 threshold",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{figure_prefix.replace('_', ' ').title()} precision / recall tradeoff")
    ax.set_xlim(float(grid["threshold"].min()), float(grid["threshold"].max()))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(precision_recall_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grid["threshold"], grid["attack_fnr"], label="Attack FNR", color="#d62728")
    ax.axvline(default_thr, color="#444444", linestyle="--", linewidth=1.5, label="Default threshold")
    ax.axvline(
        recommended["lowest_fnr_under_min_precision"]["threshold"],
        color="#2ca02c",
        linestyle=":",
        linewidth=2,
        label="Lowest FNR under precision constraint",
    )
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("FNR")
    ax.set_title(f"{figure_prefix.replace('_', ' ').title()} false-negative rate")
    ax.set_xlim(float(grid["threshold"].min()), float(grid["threshold"].max()))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fnr_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return precision_recall_path, fnr_path


def write_summary_markdown(
    *,
    holdout_recs: dict[str, object],
    cv_recs: dict[str, object] | None,
    multiclass_recs: dict[str, object] | None,
    output_path: Path,
) -> None:
    holdout_recommendations = holdout_recs["recommendations"]
    lines = [
        "# Threshold Calibration Summary",
        "",
        "This step evaluates a fixed threshold grid on the saved holdout probabilities, and on out-of-fold CV probabilities when available.",
        "PR-AUC is reported as context from the underlying probability scores; it does not change with threshold.",
        "",
        "## Holdout",
        f"- Threshold grid: {holdout_recs['threshold_grid']['min']:.2f} to {holdout_recs['threshold_grid']['max']:.2f} step {holdout_recs['threshold_grid']['step']:.2f}",
        f"- Default threshold: {holdout_recommendations['default_threshold']['threshold']:.2f}",
        f"- Max F1 threshold: {holdout_recommendations['recommended']['max_f1']['threshold']:.2f}",
        f"- Max F2 threshold: {holdout_recommendations['recommended']['max_f2']['threshold']:.2f}",
        f"- Best recall under precision constraint: {holdout_recommendations['recommended']['highest_recall_under_min_precision']['threshold']:.2f}",
        f"- Lowest FNR under precision constraint: {holdout_recommendations['recommended']['lowest_fnr_under_min_precision']['threshold']:.2f}",
        "",
    ]

    if cv_recs is not None:
        cv_recommendations = cv_recs["recommendations"]
        lines.extend(
            [
                "## CV",
                f"- Threshold grid: {cv_recs['threshold_grid']['min']:.2f} to {cv_recs['threshold_grid']['max']:.2f} step {cv_recs['threshold_grid']['step']:.2f}",
                f"- Default threshold: {cv_recommendations['default_threshold']['threshold']:.2f}",
                f"- Max F1 threshold: {cv_recommendations['recommended']['max_f1']['threshold']:.2f}",
                f"- Max F2 threshold: {cv_recommendations['recommended']['max_f2']['threshold']:.2f}",
                f"- Best recall under precision constraint: {cv_recommendations['recommended']['highest_recall_under_min_precision']['threshold']:.2f}",
                f"- Lowest FNR under precision constraint: {cv_recommendations['recommended']['lowest_fnr_under_min_precision']['threshold']:.2f}",
                "",
            ]
        )

    if multiclass_recs is not None:
        lines.extend(
            [
                "## Multiclass",
                f"- Source: {multiclass_recs['source']}",
                f"- Output CSV: {multiclass_recs['output_csv']}",
                f"- Critical classes: {', '.join(multiclass_recs.get('critical_classes', [])) or 'n/a'}",
                f"- Minimum precision: {float(multiclass_recs.get('min_precision', 0.0)):.2f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Why these thresholds are reasonable",
            "- They make the precision/recall tradeoff explicit for the binary attack detector.",
            "- The minimum-precision constraint prevents the paper from recommending thresholds that lower false alarms too aggressively at the expense of precision.",
            "- The lowest-FNR recommendation is useful when the paper emphasizes attack detection and missed-attack reduction.",
            "- The max-F1 and max-F2 thresholds provide standard reference points for the tradeoff discussion.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def calibrate_dataset(
    *,
    predictions_path: Path,
    grid_output_path: Path,
    recommendations_output_path: Path,
    figure_prefix: str,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    min_precision: float,
    default_threshold: float,
    figure_dir: Path,
) -> dict[str, object]:
    df = load_prediction_frame(predictions_path)
    thresholds = build_threshold_grid(
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )
    grid = compute_threshold_metrics(df["true_label"].to_numpy(), df["pred_proba_attack"].to_numpy(), thresholds)
    grid.to_csv(grid_output_path, index=False)

    recommendations = select_recommendations(
        grid,
        min_precision=min_precision,
        default_threshold=default_threshold,
    )
    write_recommendations(
        grid=grid,
        recommendations=recommendations,
        output_path=recommendations_output_path,
        source_name=str(predictions_path),
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )
    precision_recall_fig, fnr_fig = save_threshold_figures(
        grid=grid,
        recommendations=recommendations,
        figure_prefix=figure_prefix,
        figure_dir=figure_dir,
    )

    return {
        "source": str(predictions_path),
        "threshold_grid": {
            "min": float(threshold_min),
            "max": float(threshold_max),
            "step": float(threshold_step),
            "count": int(len(grid)),
        },
        "grid_output": str(grid_output_path),
        "recommendations_output": str(recommendations_output_path),
        "precision_recall_figure": str(precision_recall_fig),
        "fnr_figure": str(fnr_fig),
        "recommendations": recommendations,
    }


def load_multiclass_prediction_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Multiclass prediction file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = {"true_label_index", "pred_label_index", "true_label_name", "pred_label_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Multiclass prediction file {path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    for column in ["true_label_index", "pred_label_index"]:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
    return df


def calibrate_multiclass_dataset(
    *,
    predictions_path: Path,
    output_path: Path,
    default_threshold: float = 0.5,
    min_precision: float = 0.97,
    critical_classes: list[str] | None = None,
) -> dict[str, object]:
    df = load_multiclass_prediction_frame(predictions_path)
    class_columns = sorted([column[len("proba_") :] for column in df.columns if column.startswith("proba_")])
    if not class_columns:
        raise ValueError(f"No per-class probability columns found in {predictions_path}.")
    proba = np.column_stack([pd.to_numeric(df[f"proba_{cls}"], errors="coerce").fillna(0.0).to_numpy() for cls in class_columns])
    y_true = df["true_label_index"].astype(int)
    thresholds = calibrate_multiclass_thresholds(
        y_true,
        proba,
        class_columns,
        default_threshold=default_threshold,
        min_precision=min_precision,
        critical_classes=critical_classes or [],
    )
    table = thresholds["table"].copy()
    table.to_csv(output_path.with_suffix(".csv"), index=False)
    payload = {
        "source": str(predictions_path),
        "output_csv": str(output_path.with_suffix(".csv")),
        "default_threshold": float(default_threshold),
        "min_precision": float(min_precision),
        "critical_classes": critical_classes or [],
        "thresholds": thresholds["per_class"],
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Threshold calibration for the offline Edge-IIoT IDS."
    )
    parser.add_argument("--holdout_predictions", default=str(DEFAULT_HOLDOUT_PREDICTIONS))
    parser.add_argument("--cv_predictions", default=str(DEFAULT_CV_PREDICTIONS))
    parser.add_argument("--multiclass_predictions", default=None)
    parser.add_argument("--multiclass_output", default="output/reports/edge_iiot_multiclass_thresholds.json")
    parser.add_argument("--output_report_dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--output_fig_dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--min_precision", type=float, default=0.97)
    parser.add_argument("--default_threshold", type=float, default=0.5)
    parser.add_argument(
        "--critical_classes",
        nargs="*",
        default=["Ransomware", "MITM", "DDoS_HTTP", "DDoS_TCP", "DDoS_UDP", "DDoS_ICMP"],
    )
    parser.add_argument("--skip_cv", action="store_true", help="Only calibrate the holdout predictions.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        report_dir = Path(args.output_report_dir)
        figure_dir = Path(args.output_fig_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        holdout_result = calibrate_dataset(
            predictions_path=Path(args.holdout_predictions),
            grid_output_path=report_dir / "edge_iiot_holdout_threshold_grid.csv",
            recommendations_output_path=report_dir / "edge_iiot_holdout_threshold_recommendations.json",
            figure_prefix="edge_iiot_holdout",
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            min_precision=args.min_precision,
            default_threshold=args.default_threshold,
            figure_dir=figure_dir,
        )

        cv_result = None
        if not args.skip_cv:
            cv_path = Path(args.cv_predictions)
            if cv_path.exists():
                cv_result = calibrate_dataset(
                    predictions_path=cv_path,
                    grid_output_path=report_dir / "edge_iiot_cv_threshold_grid.csv",
                    recommendations_output_path=report_dir / "edge_iiot_cv_threshold_recommendations.json",
                    figure_prefix="edge_iiot_cv",
                    threshold_min=args.threshold_min,
                    threshold_max=args.threshold_max,
                    threshold_step=args.threshold_step,
                    min_precision=args.min_precision,
                    default_threshold=args.default_threshold,
                    figure_dir=figure_dir,
                )
            else:
                print(f"CV predictions file not found, skipping CV calibration: {cv_path}")

        multiclass_result = None
        if args.multiclass_predictions:
            multiclass_result = calibrate_multiclass_dataset(
                predictions_path=Path(args.multiclass_predictions),
                output_path=Path(args.multiclass_output),
                default_threshold=args.default_threshold,
                min_precision=args.min_precision,
                critical_classes=list(args.critical_classes),
            )

        summary_path = report_dir / "edge_iiot_threshold_calibration_summary.md"
        write_summary_markdown(
            holdout_recs=holdout_result,
            cv_recs=cv_result,
            multiclass_recs=multiclass_result,
            output_path=summary_path,
        )

        print("Threshold calibration complete.")
        print(f"Holdout grid         : {holdout_result['grid_output']}")
        print(f"Holdout recommendations: {holdout_result['recommendations_output']}")
        print(f"Holdout figures      : {holdout_result['precision_recall_figure']}")
        print(f"                     : {holdout_result['fnr_figure']}")
        if cv_result:
            print(f"CV grid              : {cv_result['grid_output']}")
            print(f"CV recommendations   : {cv_result['recommendations_output']}")
            print(f"CV figures           : {cv_result['precision_recall_figure']}")
            print(f"                     : {cv_result['fnr_figure']}")
        if multiclass_result:
            print(f"Multiclass thresholds: {multiclass_result['output_csv']}")
            print(f"Multiclass summary   : {args.multiclass_output}")
        print(f"Summary markdown     : {summary_path}")
        print(f"Runtime seconds      : {time.perf_counter() - started:.3f}")

        print("\nHoldout recommended thresholds:")
        for name, rec in holdout_result["recommendations"]["recommended"].items():
            print(f"{name:35s} {rec['threshold']:.2f}")
        if cv_result:
            print("\nCV recommended thresholds:")
            for name, rec in cv_result["recommendations"]["recommended"].items():
                print(f"{name:35s} {rec['threshold']:.2f}")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
