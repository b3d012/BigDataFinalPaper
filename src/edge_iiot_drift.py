from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from edge_iiot_experiment import (
    DEFAULT_EDGE_CSV,
    coerce_feature_types,
    get_transformed_feature_names,
    make_preprocessor,
    normalize_columns,
    prepare_training_frame,
)


DEFAULT_BUNDLE = Path("models/edge_iiot_xgb_model.joblib")
DEFAULT_REPORT_DIR = Path("output/reports")
DEFAULT_FIGURE_DIR = Path("output/figures")
DEFAULT_DEMO_DIR = Path("output/demo")
DEFAULT_DEMO_PREDICTIONS = DEFAULT_DEMO_DIR / "edge_iiot_demo_predictions.csv"
DEFAULT_DEMO_DRIFT_SUMMARY = DEFAULT_DEMO_DIR / "edge_iiot_demo_drift_summary.json"
DEFAULT_FEATURE_SCORES = DEFAULT_REPORT_DIR / "edge_iiot_drift_feature_scores.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_REPORT_DIR / "edge_iiot_drift_summary.json"
DEFAULT_RUN_SUMMARY = DEFAULT_REPORT_DIR / "edge_iiot_drift_run_summary.md"
DEFAULT_FIGURE = DEFAULT_FIGURE_DIR / "edge_iiot_drift_top_features.png"

DEFAULT_N_BINS = 10
DEFAULT_TOP_N = 15
DEFAULT_LOW_THRESHOLD = 0.10
DEFAULT_HIGH_THRESHOLD = 0.25


def load_bundle(model_path: Path) -> dict[str, object]:
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier bundle not found: {model_path}")
    bundle = joblib.load(model_path)
    required = {"preprocessor", "training_meta"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Classifier bundle is missing required keys: {sorted(missing)}")
    return bundle


def prepare_feature_frame(
    df: pd.DataFrame,
    training_meta: dict[str, object],
    *,
    numeric_threshold: float,
) -> pd.DataFrame:
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
        numeric_threshold=numeric_threshold,
    )
    return typed[feature_columns]


def to_dense_column(matrix, column_idx: int) -> np.ndarray:
    if hasattr(matrix, "getcol"):
        return np.asarray(matrix.getcol(column_idx).toarray()).ravel()
    return np.asarray(matrix[:, column_idx]).ravel()


def safe_probabilities(counts: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if counts.sum() <= 0:
        return np.full_like(counts, 1.0 / max(len(counts), 1), dtype=float)
    probs = counts / counts.sum()
    probs = np.clip(probs, epsilon, None)
    probs = probs / probs.sum()
    return probs


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    m = np.clip(m, 1e-12, None)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def compute_feature_drift(
    ref_values: np.ndarray,
    tgt_values: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, float]:
    ref_values = np.asarray(ref_values, dtype=float)
    tgt_values = np.asarray(tgt_values, dtype=float)

    ref_values = ref_values[np.isfinite(ref_values)]
    tgt_values = tgt_values[np.isfinite(tgt_values)]
    if ref_values.size == 0 or tgt_values.size == 0:
        return {
            "psi": 0.0,
            "js_divergence": 0.0,
            "reference_mean": float(np.mean(ref_values)) if ref_values.size else 0.0,
            "target_mean": float(np.mean(tgt_values)) if tgt_values.size else 0.0,
            "abs_mean_shift": 0.0,
            "reference_std": float(np.std(ref_values)) if ref_values.size else 0.0,
            "target_std": float(np.std(tgt_values)) if tgt_values.size else 0.0,
            "reference_min": float(np.min(ref_values)) if ref_values.size else 0.0,
            "reference_max": float(np.max(ref_values)) if ref_values.size else 0.0,
            "target_min": float(np.min(tgt_values)) if tgt_values.size else 0.0,
            "target_max": float(np.max(tgt_values)) if tgt_values.size else 0.0,
            "reference_nonzero_rate": float(np.mean(ref_values != 0)) if ref_values.size else 0.0,
            "target_nonzero_rate": float(np.mean(tgt_values != 0)) if tgt_values.size else 0.0,
            "bin_count": 0,
        }

    ref_unique = np.unique(ref_values)
    tgt_unique = np.unique(tgt_values)
    if ref_unique.size == 1 and tgt_unique.size == 1 and np.isclose(ref_unique[0], tgt_unique[0]):
        return {
            "psi": 0.0,
            "js_divergence": 0.0,
            "reference_mean": float(np.mean(ref_values)),
            "target_mean": float(np.mean(tgt_values)),
            "abs_mean_shift": float(abs(np.mean(ref_values) - np.mean(tgt_values))),
            "reference_std": float(np.std(ref_values)),
            "target_std": float(np.std(tgt_values)),
            "reference_min": float(np.min(ref_values)),
            "reference_max": float(np.max(ref_values)),
            "target_min": float(np.min(tgt_values)),
            "target_max": float(np.max(tgt_values)),
            "reference_nonzero_rate": float(np.mean(ref_values != 0)),
            "target_nonzero_rate": float(np.mean(tgt_values != 0)),
            "bin_count": 1,
        }

    edges = np.unique(np.quantile(ref_values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        lo = float(min(np.min(ref_values), np.min(tgt_values)))
        hi = float(max(np.max(ref_values), np.max(tgt_values)))
        if math.isclose(lo, hi):
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
    if np.unique(edges).size < 3:
        edges = np.linspace(float(np.min(ref_values)), float(np.max(ref_values)) + 1.0, n_bins + 1)

    ref_hist, _ = np.histogram(ref_values, bins=edges)
    tgt_hist, _ = np.histogram(tgt_values, bins=edges)
    ref_pct = safe_probabilities(ref_hist)
    tgt_pct = safe_probabilities(tgt_hist)

    psi = float(np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct)))
    js = float(jensen_shannon_divergence(ref_pct, tgt_pct))

    return {
        "psi": psi,
        "js_divergence": js,
        "reference_mean": float(np.mean(ref_values)),
        "target_mean": float(np.mean(tgt_values)),
        "abs_mean_shift": float(abs(np.mean(ref_values) - np.mean(tgt_values))),
        "reference_std": float(np.std(ref_values)),
        "target_std": float(np.std(tgt_values)),
        "reference_min": float(np.min(ref_values)),
        "reference_max": float(np.max(ref_values)),
        "target_min": float(np.min(tgt_values)),
        "target_max": float(np.max(tgt_values)),
        "reference_nonzero_rate": float(np.mean(ref_values != 0)),
        "target_nonzero_rate": float(np.mean(tgt_values != 0)),
        "bin_count": int(len(edges) - 1),
    }


def severity_from_psi(psi_values: pd.Series) -> tuple[str, bool, dict[str, int | float]]:
    max_psi = float(psi_values.max()) if not psi_values.empty else 0.0
    mean_psi = float(psi_values.mean()) if not psi_values.empty else 0.0
    median_psi = float(psi_values.median()) if not psi_values.empty else 0.0
    count_high = int((psi_values >= DEFAULT_HIGH_THRESHOLD).sum())
    count_moderate = int((psi_values >= DEFAULT_LOW_THRESHOLD).sum())

    if max_psi >= DEFAULT_HIGH_THRESHOLD:
        severity = "high"
    elif max_psi >= DEFAULT_LOW_THRESHOLD:
        severity = "moderate"
    else:
        severity = "low"

    return severity, severity != "low", {
        "max_psi": max_psi,
        "mean_psi": mean_psi,
        "median_psi": median_psi,
        "features_psi_ge_0_10": count_moderate,
        "features_psi_ge_0_25": count_high,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def compare_batches(
    ref_raw: pd.DataFrame,
    tgt_raw: pd.DataFrame,
    *,
    bundle: dict[str, object],
    mode: str,
    reference_name: str,
    target_name: str,
    n_bins: int,
    min_category_count: int,
    numeric_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    training_meta = bundle["training_meta"]
    feature_columns = list(training_meta["feature_columns"])

    ref_X = prepare_feature_frame(ref_raw, training_meta, numeric_threshold=numeric_threshold)[feature_columns]
    tgt_X = prepare_feature_frame(tgt_raw, training_meta, numeric_threshold=numeric_threshold)[feature_columns]

    preprocessor = make_preprocessor(
        training_meta["numeric_columns"],
        training_meta["categorical_columns"],
        min_category_count=min_category_count,
    )
    preprocessor.fit(ref_X)
    ref_tx = preprocessor.transform(ref_X)
    tgt_tx = preprocessor.transform(tgt_X)

    feature_names = get_transformed_feature_names(preprocessor)
    if len(feature_names) != getattr(ref_tx, "shape", (0, 0))[1]:
        feature_names = feature_names[: ref_tx.shape[1]]

    rows: list[dict[str, float | str | int]] = []
    for idx, feature_name in enumerate(feature_names):
        ref_values = to_dense_column(ref_tx, idx)
        tgt_values = to_dense_column(tgt_tx, idx)
        drift = compute_feature_drift(ref_values, tgt_values, n_bins=n_bins)
        row = {
            "feature": feature_name,
            "psi": drift["psi"],
            "js_divergence": drift["js_divergence"],
            "reference_mean": drift["reference_mean"],
            "target_mean": drift["target_mean"],
            "abs_mean_shift": drift["abs_mean_shift"],
            "reference_std": drift["reference_std"],
            "target_std": drift["target_std"],
            "reference_min": drift["reference_min"],
            "reference_max": drift["reference_max"],
            "target_min": drift["target_min"],
            "target_max": drift["target_max"],
            "reference_nonzero_rate": drift["reference_nonzero_rate"],
            "target_nonzero_rate": drift["target_nonzero_rate"],
            "bin_count": drift["bin_count"],
        }
        rows.append(row)

    feature_scores = pd.DataFrame(rows).sort_values("psi", ascending=False, kind="mergesort").reset_index(drop=True)
    feature_scores.insert(0, "rank", np.arange(1, len(feature_scores) + 1))

    severity, drift_flag, severity_stats = severity_from_psi(feature_scores["psi"])
    top_features = feature_scores.head(DEFAULT_TOP_N).to_dict(orient="records")
    summary = {
        "mode": mode,
        "reference_name": reference_name,
        "target_name": target_name,
        "reference_rows": int(len(ref_raw)),
        "target_rows": int(len(tgt_raw)),
        "feature_count": int(len(feature_scores)),
        "n_bins": int(n_bins),
        "min_category_count": int(min_category_count),
        "numeric_threshold": float(numeric_threshold),
        "drift_rule": {
            "low": f"max PSI < {DEFAULT_LOW_THRESHOLD:.2f}",
            "moderate": f"{DEFAULT_LOW_THRESHOLD:.2f} <= max PSI < {DEFAULT_HIGH_THRESHOLD:.2f}",
            "high": f"max PSI >= {DEFAULT_HIGH_THRESHOLD:.2f}",
        },
        "severity": severity,
        "drift_flag": bool(drift_flag),
        "severity_stats": severity_stats,
        "top_features": top_features,
    }

    reference_profile = ref_X.copy()
    reference_profile["batch"] = "reference"
    target_profile = tgt_X.copy()
    target_profile["batch"] = "target"
    batch_preview = pd.concat([reference_profile.head(3), target_profile.head(3)], ignore_index=True)
    return feature_scores, summary, batch_preview


def load_dataset_batches(edge_csv: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    X, y, training_meta = prepare_training_frame(
        edge_csv,
        keep_identity_payload=False,
        numeric_threshold=0.95,
        sample_rows=None,
        drop_duplicates=True,
    )
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    ref_raw = X.iloc[train_idx].reset_index(drop=True)
    tgt_raw = X.iloc[test_idx].reset_index(drop=True)
    return ref_raw, tgt_raw, training_meta


def load_demo_target_batch(demo_dir: Path, demo_predictions: Path) -> pd.DataFrame:
    if demo_predictions.exists():
        df = pd.read_csv(demo_predictions, low_memory=False)
        return normalize_columns(df)

    extracted_dir = demo_dir / "extracted_csvs"
    if extracted_dir.exists():
        frames = []
        for path in sorted(extracted_dir.glob("*.csv")):
            frames.append(normalize_columns(pd.read_csv(path, low_memory=False)))
        if frames:
            return pd.concat(frames, ignore_index=True)

    raise FileNotFoundError(
        "Could not find demo row-level data. Expected either "
        f"{demo_predictions} or CSVs in {extracted_dir}."
    )


def load_demo_reference_batch(edge_csv: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    X, y, training_meta = prepare_training_frame(
        edge_csv,
        keep_identity_payload=False,
        numeric_threshold=0.95,
        sample_rows=None,
        drop_duplicates=True,
    )
    from sklearn.model_selection import train_test_split

    train_idx, _ = train_test_split(
        np.arange(len(X)),
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    train_X = X.iloc[train_idx].reset_index(drop=True)
    train_y = y.iloc[train_idx].reset_index(drop=True)
    benign_ref = train_X[train_y == 0].reset_index(drop=True)
    return benign_ref, training_meta


def plot_top_features(feature_scores: pd.DataFrame, output_path: Path, top_n: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"Skipping drift figure because matplotlib is unavailable: {exc}")
        return

    top = feature_scores.head(top_n).iloc[::-1].copy()
    if top.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(5.0, 0.36 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.barh(top["feature"], top["psi"], color="#2c7fb8")
    ax.set_xlabel("PSI")
    ax.set_title("Top Drifted Features")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_outputs(
    *,
    feature_scores: pd.DataFrame,
    summary: dict[str, object],
    mode: str,
    report_dir: Path,
    figure_dir: Path,
    demo_dir: Path | None = None,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if demo_dir is not None:
        demo_dir.mkdir(parents=True, exist_ok=True)

    feature_scores_path = DEFAULT_FEATURE_SCORES
    summary_path = DEFAULT_SUMMARY_JSON
    run_summary_path = DEFAULT_RUN_SUMMARY
    figure_path = DEFAULT_FIGURE

    feature_scores.to_csv(feature_scores_path, index=False)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(json_safe(summary), fh, indent=2)
    plot_top_features(feature_scores, figure_path, DEFAULT_TOP_N)

    run_lines = [
        "# Edge-IIoT Drift Detection Summary",
        "",
        f"- Mode: {mode}",
        f"- Reference batch: {summary['reference_name']}",
        f"- Target batch: {summary['target_name']}",
        f"- Reference rows: {summary['reference_rows']}",
        f"- Target rows: {summary['target_rows']}",
        f"- Feature count: {summary['feature_count']}",
        f"- Overall severity: {summary['severity']}",
        f"- Drift flag: {summary['drift_flag']}",
        f"- Drift rule: max PSI < {DEFAULT_LOW_THRESHOLD:.2f} => low, "
        f"{DEFAULT_LOW_THRESHOLD:.2f} <= max PSI < {DEFAULT_HIGH_THRESHOLD:.2f} => moderate, "
        f"max PSI >= {DEFAULT_HIGH_THRESHOLD:.2f} => high",
        "",
        "## Top Drifted Features",
    ]
    for item in summary["top_features"][:DEFAULT_TOP_N]:
        run_lines.append(f"- {item['feature']}: PSI={item['psi']:.6f}, JS={item['js_divergence']:.6f}")
    run_lines.extend(["", "## Notes", "- PSI is computed on the transformed feature space produced by the saved classifier contract."])
    run_summary_path.write_text("\n".join(run_lines), encoding="utf-8")

    outputs = {
        "feature_scores": feature_scores_path,
        "summary": summary_path,
        "run_summary": run_summary_path,
        "figure": figure_path,
    }

    if mode == "demo" and demo_dir is not None:
        demo_summary_path = demo_dir / "edge_iiot_demo_drift_summary.json"
        with demo_summary_path.open("w", encoding="utf-8") as fh:
            json.dump(json_safe(summary), fh, indent=2)
        outputs["demo_summary"] = demo_summary_path

    return outputs


def run_dataset_mode(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.classifier_bundle))
    ref_raw, tgt_raw, _ = load_dataset_batches(args.edge_csv)
    feature_scores, summary, _ = compare_batches(
        ref_raw,
        tgt_raw,
        bundle=bundle,
        mode="dataset",
        reference_name="dataset_train_split",
        target_name="dataset_holdout_split",
        n_bins=args.n_bins,
        min_category_count=args.min_category_count,
        numeric_threshold=args.numeric_threshold,
    )
    outputs = save_outputs(
        feature_scores=feature_scores,
        summary=summary,
        mode="dataset",
        report_dir=Path(args.report_dir),
        figure_dir=Path(args.figure_dir),
    )
    print("Dataset drift comparison complete.")
    print(f"Severity: {summary['severity']}")
    print(f"Top drifted features: {', '.join(item['feature'] for item in summary['top_features'][:5])}")
    print("Saved outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


def run_demo_mode(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.classifier_bundle))
    ref_raw, _ = load_demo_reference_batch(args.edge_csv)
    tgt_raw = load_demo_target_batch(Path(args.demo_dir), Path(args.demo_predictions))
    feature_scores, summary, _ = compare_batches(
        ref_raw,
        tgt_raw,
        bundle=bundle,
        mode="demo",
        reference_name="dataset_train_benign_split",
        target_name="demo_replay_rows",
        n_bins=args.n_bins,
        min_category_count=args.min_category_count,
        numeric_threshold=args.numeric_threshold,
    )
    outputs = save_outputs(
        feature_scores=feature_scores,
        summary=summary,
        mode="demo",
        report_dir=Path(args.report_dir),
        figure_dir=Path(args.figure_dir),
        demo_dir=Path(args.demo_dir),
    )
    print("Demo drift comparison complete.")
    print(f"Severity: {summary['severity']}")
    print(f"Top drifted features: {', '.join(item['feature'] for item in summary['top_features'][:5])}")
    print("Saved outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Edge-IIoT drift detection on transformed feature batches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", help="Compare train/holdout splits from the Edge-IIoT dataset.")
    dataset.add_argument("--edge_csv", default=DEFAULT_EDGE_CSV)
    dataset.add_argument("--classifier_bundle", default=str(DEFAULT_BUNDLE))
    dataset.add_argument("--report_dir", default=str(DEFAULT_REPORT_DIR))
    dataset.add_argument("--figure_dir", default=str(DEFAULT_FIGURE_DIR))
    dataset.add_argument("--n_bins", type=int, default=DEFAULT_N_BINS)
    dataset.add_argument("--min_category_count", type=int, default=20)
    dataset.add_argument("--numeric_threshold", type=float, default=0.95)
    dataset.set_defaults(func=run_dataset_mode)

    demo = subparsers.add_parser("demo", help="Compare benign reference traffic with demo replay rows.")
    demo.add_argument("--edge_csv", default=DEFAULT_EDGE_CSV)
    demo.add_argument("--classifier_bundle", default=str(DEFAULT_BUNDLE))
    demo.add_argument("--demo_dir", default=str(DEFAULT_DEMO_DIR))
    demo.add_argument("--demo_predictions", default=str(DEFAULT_DEMO_PREDICTIONS))
    demo.add_argument("--report_dir", default=str(DEFAULT_REPORT_DIR))
    demo.add_argument("--figure_dir", default=str(DEFAULT_FIGURE_DIR))
    demo.add_argument("--n_bins", type=int, default=DEFAULT_N_BINS)
    demo.add_argument("--min_category_count", type=int, default=20)
    demo.add_argument("--numeric_threshold", type=float, default=0.95)
    demo.set_defaults(func=run_demo_mode)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
