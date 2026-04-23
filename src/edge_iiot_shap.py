from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from edge_iiot_experiment import (
    DEFAULT_EDGE_CSV,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_DIR,
    get_transformed_feature_names,
    prepare_training_frame,
)


DEFAULT_FIGURE_DIR = Path("output/figures")
DEFAULT_GLOBAL_SAMPLE_SIZE = 1000
DEFAULT_TOP_K = 20


def load_bundle(model_path: Path) -> dict[str, object]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    bundle = joblib.load(model_path)
    required = {"model", "preprocessor", "training_meta"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {sorted(missing)}")
    return bundle


def rebuild_holdout_split(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

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
    return train_idx, val_idx, test_idx


def to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            return np.asarray(shap_values[0])
        return np.asarray(shap_values[-1])
    return np.asarray(shap_values)


def ensure_directories(report_dir: Path, figure_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)


def build_holdout_frame(
    edge_csv: Path,
    bundle: dict[str, object],
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X, y, _ = prepare_training_frame(
        edge_csv,
        keep_identity_payload=False,
        numeric_threshold=0.95,
        sample_rows=None,
        drop_duplicates=True,
    )
    _, _, test_idx = rebuild_holdout_split(X, y)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    test_tx = bundle["preprocessor"].transform(X_test)
    pred_proba = bundle["model"].predict_proba(test_tx)[:, 1]
    pred_label = (pred_proba >= threshold).astype(int)

    holdout = pd.DataFrame(
        {
            "holdout_index": np.arange(len(X_test)),
            "true_label": y_test.values,
            "true_label_name": y_test.map({0: "Normal", 1: "Attack"}).values,
            "pred_proba_attack": pred_proba,
            "pred_label": pred_label,
            "pred_label_name": pd.Series(pred_label).map({0: "Normal", 1: "Attack"}).values,
        }
    )
    return X_test, y_test, holdout


def select_global_sample(
    X_test: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    sample_size: int,
    random_state: int,
) -> pd.Index:
    sample_size = min(int(sample_size), len(X_test))
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    sample = holdout.sample(n=sample_size, random_state=random_state, replace=False)
    return sample.index.sort_values()


def select_local_examples(holdout: pd.DataFrame) -> pd.DataFrame:
    high_conf_attack = holdout.loc[holdout["pred_proba_attack"].idxmax()].copy()
    high_conf_attack["selection_type"] = "high_confidence_attack"

    attack_mask = holdout["true_label"] == 1
    attack_candidates = holdout.loc[attack_mask & (holdout["pred_proba_attack"] >= 0.5)]
    if attack_candidates.empty:
        attack_candidates = holdout.loc[attack_mask]
    borderline_attack = attack_candidates.loc[
        (attack_candidates["pred_proba_attack"] - 0.5).abs().idxmin()
    ].copy()
    borderline_attack["selection_type"] = "borderline_attack"

    high_conf_normal = holdout.loc[holdout["pred_proba_attack"].idxmin()].copy()
    high_conf_normal["selection_type"] = "high_confidence_normal"

    selected = pd.DataFrame(
        [high_conf_attack, borderline_attack, high_conf_normal]
    ).reset_index(drop=True)
    selected["example_id"] = np.arange(1, len(selected) + 1)
    return selected


def compute_shap_matrix(model, transformed_matrix, *, shap_module):
    explainer = shap_module.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_matrix)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = base_value[-1]
    return normalize_shap_values(shap_values), float(base_value)


def build_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    std_signed = shap_values.std(axis=0)
    total = float(mean_abs.sum()) if float(mean_abs.sum()) else 1.0
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "mean_shap": mean_signed,
            "std_shap": std_signed,
            "normalized_importance": mean_abs / total,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return df.reset_index(drop=True)


def build_local_table(
    selected_examples: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for example_position, example in selected_examples.reset_index(drop=True).iterrows():
        row_shap = shap_values[int(example_position)]
        order = np.argsort(np.abs(row_shap))[::-1][:top_k]
        top_features = []
        top_positive = []
        top_negative = []

        for rank, feature_idx in enumerate(order, start=1):
            value = float(row_shap[feature_idx])
            feature = feature_names[feature_idx]
            direction = "positive" if value >= 0 else "negative"
            rows.append(
                {
                    "example_id": int(example["example_id"]),
                    "selection_type": example["selection_type"],
                    "rank": rank,
                    "feature": feature,
                    "shap_value": value,
                    "abs_shap_value": abs(value),
                    "direction": direction,
                    "true_label": int(example["true_label"]),
                    "true_label_name": example["true_label_name"],
                    "pred_label": int(example["pred_label"]),
                    "pred_label_name": example["pred_label_name"],
                    "pred_proba_attack": float(example["pred_proba_attack"]),
                    "holdout_index": int(example["holdout_index"]),
                }
            )
            top_features.append(f"{feature}:{value:.6f}")
            if value >= 0:
                top_positive.append(f"{feature}:{value:.6f}")
            else:
                top_negative.append(f"{feature}:{value:.6f}")

        summary_rows.append(
            {
                "example_id": int(example["example_id"]),
                "selection_type": example["selection_type"],
                "holdout_index": int(example["holdout_index"]),
                "true_label": int(example["true_label"]),
                "true_label_name": example["true_label_name"],
                "pred_label": int(example["pred_label"]),
                "pred_label_name": example["pred_label_name"],
                "pred_proba_attack": float(example["pred_proba_attack"]),
                "top_features": " | ".join(top_features),
                "top_positive_features": " | ".join(top_positive[:5]),
                "top_negative_features": " | ".join(top_negative[:5]),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def save_global_figure(global_importance: pd.DataFrame, figure_path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    top = global_importance.head(DEFAULT_TOP_K).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2c7fb8")
    ax.set_title("Edge-IIoT SHAP Global Importance")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def run_command(args: argparse.Namespace) -> None:
    try:
        import shap
    except ImportError as exc:
        raise SystemExit(
            "SHAP is not installed. Install it with `pip install shap` and rerun this command."
        ) from exc

    model_path = Path(args.model_path)
    edge_csv = Path(args.edge_csv)
    report_dir = Path(args.output_report_dir)
    figure_dir = Path(args.output_fig_dir)
    ensure_directories(report_dir, figure_dir)

    bundle = load_bundle(model_path)
    threshold = float(args.threshold if args.threshold is not None else bundle.get("threshold", 0.5))

    started = time.perf_counter()
    X_test, y_test, holdout = build_holdout_frame(edge_csv, bundle, threshold=threshold)
    global_sample_idx = select_global_sample(
        X_test,
        holdout,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    X_global = X_test.iloc[global_sample_idx].reset_index(drop=True)
    X_global_tx = bundle["preprocessor"].transform(X_global)
    global_dense = to_dense(X_global_tx)

    feature_names = get_transformed_feature_names(bundle["preprocessor"])
    booster = bundle["model"].get_booster()

    shap_values_global, base_value = compute_shap_matrix(
        booster,
        global_dense,
        shap_module=shap,
    )
    global_importance = build_global_importance(shap_values_global, feature_names)
    top_features = global_importance.head(args.top_k).copy()
    selected_examples = select_local_examples(holdout)

    local_rows = []
    local_summary_rows = []
    for _, selected in selected_examples.iterrows():
        row_index = int(selected["holdout_index"])
        X_one = X_test.iloc[[row_index]].reset_index(drop=True)
        X_one_tx = to_dense(bundle["preprocessor"].transform(X_one))
        local_shap, _local_base_value = compute_shap_matrix(
            booster,
            X_one_tx,
            shap_module=shap,
        )
        local_table, local_summary = build_local_table(
            selected_examples.loc[[selected.name]],
            local_shap,
            feature_names,
            top_k=args.local_top_k,
        )
        local_rows.append(local_table)
        local_summary_rows.append(local_summary)

    local_examples = pd.concat(local_rows, ignore_index=True)
    local_summary = pd.concat(local_summary_rows, ignore_index=True)

    global_importance_path = report_dir / "edge_iiot_shap_global_importance.csv"
    top_features_path = report_dir / "edge_iiot_shap_top_features.csv"
    local_examples_path = report_dir / "edge_iiot_shap_local_examples.csv"
    local_summary_path = report_dir / "edge_iiot_shap_local_summary.csv"
    summary_json_path = report_dir / "edge_iiot_shap_summary.json"
    figure_path = figure_dir / "edge_iiot_shap_summary.png"

    global_importance.to_csv(global_importance_path, index=False)
    top_features.to_csv(top_features_path, index=False)
    local_examples.to_csv(local_examples_path, index=False)
    local_summary.to_csv(local_summary_path, index=False)
    saved_figure = save_global_figure(global_importance, figure_path)

    summary = {
        "model_path": str(model_path),
        "edge_csv": str(edge_csv),
        "threshold": threshold,
        "global_sample_size": int(len(global_sample_idx)),
        "local_example_count": int(len(selected_examples)),
        "global_explanation_type": "mean_absolute_shap_on_holdout_sample",
        "local_explanation_type": "holdout_example_shap_tables",
        "random_state": int(args.random_state),
        "top_features": top_features["feature"].tolist(),
        "top_feature_mean_abs_shap": top_features["mean_abs_shap"].round(8).tolist(),
        "summary_figure": str(saved_figure) if saved_figure else None,
        "runtime_seconds": float(time.perf_counter() - started),
        "base_value": float(base_value),
    }
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("SHAP explainability complete.")
    print(f"Global importance : {global_importance_path}")
    print(f"Top features      : {top_features_path}")
    print(f"Local explanations : {local_examples_path}")
    print(f"Local summary     : {local_summary_path}")
    print(f"Summary JSON      : {summary_json_path}")
    if saved_figure:
        print(f"Summary figure    : {saved_figure}")
    print("\nTop SHAP features:")
    print(top_features[["feature", "mean_abs_shap"]].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SHAP explainability for the offline Edge-IIoT XGBoost classifier."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--edge_csv", default=DEFAULT_EDGE_CSV)
    parser.add_argument("--output_report_dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--output_fig_dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--sample_size", type=int, default=DEFAULT_GLOBAL_SAMPLE_SIZE)
    parser.add_argument("--local_top_k", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_command(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
