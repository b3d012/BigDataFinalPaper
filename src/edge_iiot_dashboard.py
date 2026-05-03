from __future__ import annotations

import time
import subprocess
import sys
import json
import shutil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from edge_iiot_mongo import (
    ANALYSIS_SUMMARIES_COLLECTION,
    ALERT_EXPLANATIONS_COLLECTION,
    ADVERSARIAL_EVALUATIONS_COLLECTION,
    DEFAULT_MONGO_DB,
    DEFAULT_MONGO_URI,
    DRIFT_EVENTS_COLLECTION,
    FEATURE_IMPORTANCE_COLLECTION,
    LIVE_WINDOWS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RETRAIN_EVENTS_COLLECTION,
    THRESHOLD_METRICS_COLLECTION,
    collection_counts,
    clear_live_collections,
    collection_to_dataframe,
    connect_database,
    upsert_documents,
)
from edge_iiot_experiment import evaluation_from_predictions
from edge_iiot_demo_replay import find_tshark, validate_tshark_interface
from edge_iiot_thresholds import build_threshold_grid, compute_threshold_metrics, select_recommendations
from edge_iiot_runtime import DEFAULT_ACTIVE_MODEL_POINTER_PATH


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "output" / "reports"
DEMO_DIR = REPO_ROOT / "output" / "demo"
FIGURE_DIR = REPO_ROOT / "output" / "figures"
LIVE_WORKER = REPO_ROOT / "src" / "edge_iiot_live_capture.py"
LIVE_INBOX_DIR = REPO_ROOT / "output" / "live" / "inbox"
LIVE_PROCESSED_DIR = REPO_ROOT / "output" / "live" / "processed"


def resolve_live_model_reference() -> Path:
    pointer_path = DEFAULT_ACTIVE_MODEL_POINTER_PATH
    if pointer_path.exists():
        return pointer_path
    return REPO_ROOT / "models" / "edge_iiot_xgb_model.joblib"


CLASSIFIER_SOURCE_DEFAULTS = ["holdout", "cv", "demo", "live"]
ANOMALY_SOURCE_DEFAULTS = ["holdout", "demo", "live"]
LIVE_PREDICTION_KIND = "live_prediction"
LIVE_ANOMALY_PREDICTION_KIND = "live_anomaly_prediction"
LIVE_SHAP_IMPORTANCE_KIND = "live_shap"
LIVE_DRIFT_IMPORTANCE_KIND = "live_drift"
FRIENDLY_LABELS = {
    LIVE_PREDICTION_KIND: "Classifier (Live)",
    LIVE_ANOMALY_PREDICTION_KIND: "Anomaly (Live)",
    LIVE_SHAP_IMPORTANCE_KIND: "Shap (Live)",
    LIVE_DRIFT_IMPORTANCE_KIND: "Drift (Live)",
    "live": "Live",
    "holdout": "Holdout",
    "cv": "CV",
    "demo": "Demo",
    "classifier": "Classifier",
    "anomaly": "Anomaly",
}


def friendly_label(value: str) -> str:
    return FRIENDLY_LABELS.get(value, value.replace("_", " ").title())


def infer_binary_label_from_name(name: str | None) -> int | None:
    if not name:
        return None
    lowered = str(name).lower()
    benign_tokens = ("noattack", "no_attack", "benign", "normal", "baseline")
    attack_tokens = (
        "attack",
        "flood",
        "scan",
        "bruteforce",
        "brute_force",
        "brute",
        "exploit",
        "dos",
        "ddos",
        "botnet",
        "malware",
        "syn",
    )
    if any(token in lowered for token in benign_tokens):
        return 0
    if any(token in lowered for token in attack_tokens):
        return 1
    return None


def safe_read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if path is None or not path.exists() or path.is_dir():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def metric_from_summary(summary: dict[str, object], key: str, default: float = 0.0) -> float:
    for container_key in ("evaluation_metrics", "metrics", "original_metrics", "retrained_metrics"):
        container = summary.get(container_key)
        if isinstance(container, dict) and key in container:
            value = container.get(key)
            if value is not None:
                return float(value)
    if key in summary and summary.get(key) is not None:
        return float(summary.get(key))
    return float(default)


def load_local_summary(kind: str) -> dict[str, object]:
    mapping = {
        "holdout_metrics": REPORT_DIR / "edge_iiot_holdout_metrics.json",
        "cv_summary": REPORT_DIR / "edge_iiot_cv_summary.json",
        "shap_summary": REPORT_DIR / "edge_iiot_shap_summary.json",
        "anomaly_holdout_metrics": REPORT_DIR / "edge_iiot_anomaly_holdout_metrics.json",
        "drift_summary": REPORT_DIR / "edge_iiot_drift_summary.json",
        "retrain_trigger": REPORT_DIR / "edge_iiot_retrain_trigger.json",
        "retrain_comparison": REPORT_DIR / "edge_iiot_retrain_comparison.json",
        "threshold_calibration_summary": REPORT_DIR / "edge_iiot_threshold_calibration_summary.md",
        "anomaly_run_summary": REPORT_DIR / "edge_iiot_anomaly_run_summary.md",
        "drift_run_summary": REPORT_DIR / "edge_iiot_drift_run_summary.md",
        "retrain_run_summary": REPORT_DIR / "edge_iiot_retrain_run_summary.md",
        "demo_run_summary": DEMO_DIR / "edge_iiot_demo_run_summary.md",
    }
    path = mapping.get(kind)
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return safe_read_json(path)
    return {"text": path.read_text(encoding="utf-8")}


def load_local_predictions(kind: str, source: str) -> pd.DataFrame:
    mapping = {
        ("classifier_prediction", "holdout"): REPORT_DIR / "edge_iiot_holdout_predictions.csv",
        ("classifier_prediction", "cv"): REPORT_DIR / "edge_iiot_cv_predictions.csv",
        ("classifier_prediction", "demo"): DEMO_DIR / "edge_iiot_demo_predictions.csv",
        ("classifier_prediction", "live"): pd.DataFrame(),
        ("anomaly_prediction", "holdout"): REPORT_DIR / "edge_iiot_anomaly_holdout_predictions.csv",
        ("anomaly_prediction", "demo"): DEMO_DIR / "edge_iiot_demo_anomaly_predictions.csv",
        ("anomaly_prediction", "live"): pd.DataFrame(),
    }
    value = mapping.get((kind, source), None)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return safe_read_csv(value)


def load_local_feature_importance(kind: str) -> pd.DataFrame:
    mapping = {
        "shap": REPORT_DIR / "edge_iiot_shap_global_importance.csv",
        "live_shap": REPORT_DIR / "edge_iiot_shap_global_importance.csv",
        "model_feature_importance": REPO_ROOT / "models" / "edge_iiot_xgb_model.feature_importance.csv",
        "drift": REPORT_DIR / "edge_iiot_drift_feature_scores.csv",
        "live_drift": REPORT_DIR / "edge_iiot_drift_feature_scores.csv",
    }
    return safe_read_csv(mapping.get(kind, None))


def load_local_threshold_grid(source: str) -> pd.DataFrame:
    mapping = {
        "holdout": REPORT_DIR / "edge_iiot_holdout_threshold_grid.csv",
        "cv": REPORT_DIR / "edge_iiot_cv_threshold_grid.csv",
    }
    return safe_read_csv(mapping.get(source, None))


def available_prediction_sources(mongo_uri: str, db_name: str, kind: str) -> list[str]:
    try:
        db = connect_database(mongo_uri, db_name)
        values = db[PREDICTIONS_COLLECTION].distinct("source", {"kind": kind})
        values = sorted(value for value in values if value is not None)
        if values:
            if kind in {"classifier_prediction", "anomaly_prediction"} and "live" not in values:
                values.append("live")
            return values
    except Exception:
        pass
    return CLASSIFIER_SOURCE_DEFAULTS if kind == "classifier_prediction" else ANOMALY_SOURCE_DEFAULTS


@st.cache_data(show_spinner=False, ttl=5)
def load_prediction_frame(
    mongo_uri: str,
    db_name: str,
    kind: str,
    source: str,
) -> pd.DataFrame:
    if source == "live":
        live_kind = LIVE_PREDICTION_KIND if kind == "classifier_prediction" else LIVE_ANOMALY_PREDICTION_KIND
        live_frame = load_live_prediction_frame(mongo_uri, db_name, kind=live_kind, source=None, window_id=None, limit=5000)
        if not live_frame.empty:
            return live_frame
        return pd.DataFrame()
    try:
        db = connect_database(mongo_uri, db_name)
        query = {"kind": kind, "source": source}
        df = collection_to_dataframe(
            db[PREDICTIONS_COLLECTION],
            query=query,
            projection={"_id": 0},
            sort=[("record_index", 1), ("created_at", 1)],
        )
        if not df.empty:
            return df
    except Exception:
        pass
    return load_local_predictions(kind, source)


@st.cache_data(show_spinner=False, ttl=5)
def load_feature_importance_frame(
    mongo_uri: str,
    db_name: str,
    importance_type: str,
) -> pd.DataFrame:
    try:
        db = connect_database(mongo_uri, db_name)
        df = collection_to_dataframe(
            db[FEATURE_IMPORTANCE_COLLECTION],
            query={"importance_type": importance_type},
            projection={"_id": 0},
            sort=[("normalized_importance", -1), ("mean_abs_shap", -1)],
        )
        if not df.empty:
            return df
    except Exception:
        pass
    return load_local_feature_importance(importance_type)


@st.cache_data(show_spinner=False, ttl=5)
def load_threshold_frame(
    mongo_uri: str,
    db_name: str,
    source: str,
) -> pd.DataFrame:
    try:
        db = connect_database(mongo_uri, db_name)
        if source == "live":
            live_df = collection_to_dataframe(
                db[THRESHOLD_METRICS_COLLECTION],
                query={"source": "live"},
                projection={"_id": 0},
                sort=[("threshold", 1)],
            )
            if not live_df.empty:
                try:
                    recommendations = select_recommendations(live_df, min_precision=0.97, default_threshold=0.5)
                    sample_count = int(live_df["sample_count"].iloc[0]) if "sample_count" in live_df.columns and not live_df.empty else len(live_df)
                    summary_doc = {
                        "kind": "threshold_calibration_summary",
                        "source": "live",
                        "artifact_path": "mongo://live_threshold_metrics",
                        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "payload": {
                            "text": "\n".join(
                                [
                                    "# Threshold Calibration Summary",
                                    "",
                                    "Live threshold calibration is computed from labeled live rows currently available in MongoDB.",
                                    f"- Samples: {sample_count}",
                                    f"- Threshold grid: 0.05 to 0.95 step 0.01",
                                    f"- Default threshold: {recommendations['default_threshold']['threshold']:.2f}",
                                    f"- Max F1 threshold: {recommendations['recommended']['max_f1']['threshold']:.2f}",
                                    f"- Max F2 threshold: {recommendations['recommended']['max_f2']['threshold']:.2f}",
                                    f"- Best recall under precision constraint: {recommendations['recommended']['highest_recall_under_min_precision']['threshold']:.2f}",
                                    f"- Lowest FNR under precision constraint: {recommendations['recommended']['lowest_fnr_under_min_precision']['threshold']:.2f}",
                                ]
                            ),
                            "source_name": "live",
                            "sample_count": sample_count,
                            "recommendations": recommendations,
                        },
                    }
                    upsert_documents(db[ANALYSIS_SUMMARIES_COLLECTION], [summary_doc], ["kind", "artifact_path"])
                except Exception:
                    pass
                return live_df
            live_predictions = collection_to_dataframe(
                db[PREDICTIONS_COLLECTION],
                query={"kind": LIVE_PREDICTION_KIND},
                projection={"_id": 0},
                sort=[("window_id", 1), ("record_index", 1), ("created_at", 1)],
            )
            if live_predictions.empty or "pred_proba_attack" not in live_predictions.columns:
                return pd.DataFrame()
            if "true_label" not in live_predictions.columns:
                live_predictions["true_label"] = None
            if "source_file" in live_predictions.columns:
                inferred_labels = [infer_binary_label_from_name(str(value)) for value in live_predictions["source_file"].fillna("")]
            else:
                inferred_labels = [None] * len(live_predictions)
            derived_true_label = []
            for existing, inferred in zip(live_predictions["true_label"].tolist(), inferred_labels):
                if existing is not None and existing == existing:
                    try:
                        derived_true_label.append(int(existing))
                    except Exception:
                        derived_true_label.append(inferred)
                else:
                    derived_true_label.append(inferred)
            live_predictions = live_predictions.copy()
            live_predictions["true_label"] = derived_true_label
            live_predictions = live_predictions[live_predictions["true_label"].notna()].copy()
            if live_predictions.empty:
                return pd.DataFrame()
            live_predictions["true_label"] = pd.to_numeric(live_predictions["true_label"], errors="coerce")
            live_predictions["pred_proba_attack"] = pd.to_numeric(live_predictions["pred_proba_attack"], errors="coerce")
            live_predictions = live_predictions.dropna(subset=["true_label", "pred_proba_attack"])
            if live_predictions.empty:
                return pd.DataFrame()
            thresholds = build_threshold_grid(threshold_min=0.05, threshold_max=0.95, threshold_step=0.01)
            derived_grid = compute_threshold_metrics(
                live_predictions["true_label"].astype(int).to_numpy(),
                live_predictions["pred_proba_attack"].astype(float).to_numpy(),
                thresholds,
            )
            if not derived_grid.empty:
                derived_grid["kind"] = "threshold_grid"
                derived_grid["source"] = "live"
                derived_grid["source_name"] = "live"
                derived_grid["sample_count"] = int(len(live_predictions))
                derived_grid["label_source"] = "inferred_from_source_file"
                try:
                    upsert_documents(db[THRESHOLD_METRICS_COLLECTION], derived_grid.to_dict(orient="records"), ["kind", "source", "threshold"])
                    recommendations = select_recommendations(derived_grid, min_precision=0.97, default_threshold=0.5)
                    summary_doc = {
                        "kind": "threshold_calibration_summary",
                        "source": "live",
                        "artifact_path": "mongo://live_threshold_metrics",
                        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "payload": {
                            "text": "\n".join(
                                [
                                    "# Threshold Calibration Summary",
                                    "",
                                    "Live threshold calibration is computed from labeled live rows currently available in MongoDB.",
                                    f"- Samples: {int(derived_grid['sample_count'].iloc[0]) if 'sample_count' in derived_grid.columns and not derived_grid.empty else len(live_predictions)}",
                                    f"- Threshold grid: 0.05 to 0.95 step 0.01",
                                    f"- Default threshold: {recommendations['default_threshold']['threshold']:.2f}",
                                    f"- Max F1 threshold: {recommendations['recommended']['max_f1']['threshold']:.2f}",
                                    f"- Max F2 threshold: {recommendations['recommended']['max_f2']['threshold']:.2f}",
                                    f"- Best recall under precision constraint: {recommendations['recommended']['highest_recall_under_min_precision']['threshold']:.2f}",
                                    f"- Lowest FNR under precision constraint: {recommendations['recommended']['lowest_fnr_under_min_precision']['threshold']:.2f}",
                                ]
                            ),
                            "source_name": "live",
                            "sample_count": int(derived_grid["sample_count"].iloc[0]) if "sample_count" in derived_grid.columns and not derived_grid.empty else len(live_predictions),
                            "recommendations": recommendations,
                        },
                    }
                    upsert_documents(db[ANALYSIS_SUMMARIES_COLLECTION], [summary_doc], ["kind", "artifact_path"])
                except Exception:
                    pass
            return derived_grid
        df = collection_to_dataframe(
            db[THRESHOLD_METRICS_COLLECTION],
            query={"source": source},
            projection={"_id": 0},
            sort=[("threshold", 1)],
        )
        if not df.empty:
            return df
    except Exception:
        pass
    if source == "live":
        return pd.DataFrame()
    return load_local_threshold_grid(source)


@st.cache_data(show_spinner=False, ttl=5)
def load_live_window_frame(
    mongo_uri: str,
    db_name: str,
    source: str | None = None,
    *,
    limit: int = 50,
) -> pd.DataFrame:
    try:
        db = connect_database(mongo_uri, db_name)
        query = {"kind": "live_window_summary"}
        if source:
            query["source"] = source
        df = collection_to_dataframe(
            db[LIVE_WINDOWS_COLLECTION],
            query=query,
            projection={"_id": 0},
            sort=[("window_id", -1), ("created_at", -1)],
            limit=limit,
        )
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=5)
def load_live_window_count(
    mongo_uri: str,
    db_name: str,
    source: str | None = None,
) -> int:
    try:
        db = connect_database(mongo_uri, db_name)
        query = {"kind": "live_window_summary"}
        if source:
            query["source"] = source
        return int(db[LIVE_WINDOWS_COLLECTION].count_documents(query))
    except Exception:
        return 0


def load_live_status() -> dict[str, object]:
    status_path = REPO_ROOT / "output" / "live" / "live_capture_status.json"
    return safe_read_json(status_path)


@st.cache_data(show_spinner=False, ttl=5)
def load_live_prediction_frame(
    mongo_uri: str,
    db_name: str,
    kind: str = LIVE_PREDICTION_KIND,
    source: str | None = None,
    *,
    window_id: int | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    try:
        db = connect_database(mongo_uri, db_name)
        query = {"kind": kind}
        if source:
            query["source"] = source
        if window_id is not None:
            query["window_id"] = window_id
        df = collection_to_dataframe(
            db[PREDICTIONS_COLLECTION],
            query=query,
            projection={"_id": 0},
            sort=[("window_id", -1), ("record_index", 1), ("created_at", -1)],
            limit=limit,
        )
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=5)
def load_live_alert_frame(
    mongo_uri: str,
    db_name: str,
    *,
    window_id: int | None = None,
    limit: int = 250,
) -> pd.DataFrame:
    try:
        db = connect_database(mongo_uri, db_name)
        query = {"kind": "live_alert_explanation"}
        if window_id is not None:
            query["window_id"] = window_id
        df = collection_to_dataframe(
            db[ALERT_EXPLANATIONS_COLLECTION],
            query=query,
            projection={"_id": 0},
            sort=[("window_id", -1), ("record_index", 1), ("created_at", -1)],
            limit=limit,
        )
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=5)
def load_drift_event_frame(
    mongo_uri: str,
    db_name: str,
    *,
    limit: int = 250,
) -> pd.DataFrame:
    try:
        db = connect_database(mongo_uri, db_name)
        df = collection_to_dataframe(
            db[DRIFT_EVENTS_COLLECTION],
            query={"kind": "live_drift_event"},
            projection={"_id": 0},
            sort=[("created_at", -1), ("window_id", -1)],
            limit=limit,
        )
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=5)
def load_retrain_event_frame(
    mongo_uri: str,
    db_name: str,
    *,
    limit: int = 100,
) -> pd.DataFrame:
    try:
        db = connect_database(mongo_uri, db_name)
        df = collection_to_dataframe(
            db[RETRAIN_EVENTS_COLLECTION],
            query={"kind": {"$in": ["live_retrain_request", "live_retrain_result", "offline_retrain_result"]}},
            projection={"_id": 0},
            sort=[("created_at", -1), ("window_id", -1)],
            limit=limit,
        )
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=5)
def load_adversarial_frame(
    mongo_uri: str,
    db_name: str,
    *,
    limit: int = 100,
) -> pd.DataFrame:
    try:
        db = connect_database(mongo_uri, db_name)
        df = collection_to_dataframe(
            db[ADVERSARIAL_EVALUATIONS_COLLECTION],
            projection={"_id": 0},
            sort=[("created_at", -1)],
            limit=limit,
        )
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=5)
def load_live_shap_frame(
    mongo_uri: str,
    db_name: str,
    source: str | None = None,
    *,
    window_id: int | None = None,
    limit: int = 50,
) -> pd.DataFrame:
    try:
        db = connect_database(mongo_uri, db_name)
        query = {"importance_type": LIVE_SHAP_IMPORTANCE_KIND}
        if source:
            query["source"] = source
        if window_id is not None:
            query["window_id"] = window_id
        df = collection_to_dataframe(
            db[FEATURE_IMPORTANCE_COLLECTION],
            query=query,
            projection={"_id": 0},
            sort=[("window_id", -1), ("normalized_importance", -1), ("mean_abs_contrib", -1)],
            limit=limit,
        )
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


def live_running() -> bool:
    proc = st.session_state.get("live_proc")
    return bool(proc and getattr(proc, "poll", lambda: 1)() is None)


def live_status_text() -> str:
    proc = st.session_state.get("live_proc")
    if proc is None:
        return "stopped"
    return "running" if proc.poll() is None else f"stopped (exit={proc.returncode})"


def build_live_command(
    *,
    mongo_uri: str,
    db_name: str,
    interface: str,
    tshark: str,
    model_path: str,
    model_pointer_path: str,
    window_seconds: int,
    packet_count: int | None,
    capture_filter: str | None,
    display_filter: str | None,
    threshold: float | None,
    file_max_threshold: float,
    file_ratio_threshold: float,
    min_records: int,
    shap_sample_rows: int,
    shap_top_n: int,
    no_metadata: bool,
    max_windows: int,
    pause_seconds: float,
    auto_retrain: bool,
    output_dir: str,
    capture_dir: str,
    injection_dir: str,
    processed_dir: str,
    status_path: str,
) -> list[str]:
    command = [
        sys.executable,
        str(LIVE_WORKER),
        "live",
        "--model_path",
        model_path,
        "--model_pointer_path",
        model_pointer_path,
        "--interface",
        interface,
        "--tshark",
        tshark,
        "--window_seconds",
        str(window_seconds),
        "--mongo_uri",
        mongo_uri,
        "--db_name",
        db_name,
        "--output_dir",
        output_dir,
        "--capture_dir",
        capture_dir,
        "--injection_dir",
        injection_dir,
        "--processed_dir",
        processed_dir,
        "--status_path",
        status_path,
        "--file_max_threshold",
        str(file_max_threshold),
        "--file_ratio_threshold",
        str(file_ratio_threshold),
        "--min_records",
        str(min_records),
        "--shap_sample_rows",
        str(shap_sample_rows),
        "--shap_top_n",
        str(shap_top_n),
        "--max_windows",
        str(max_windows),
        "--pause_seconds",
        str(pause_seconds),
    ]
    if packet_count is not None and packet_count > 0:
        command.extend(["--packet_count", str(packet_count)])
    if capture_filter:
        command.extend(["--capture_filter", capture_filter])
    if display_filter:
        command.extend(["--display_filter", display_filter])
    if threshold is not None:
        command.extend(["--threshold", str(threshold)])
    if no_metadata:
        command.append("--no_metadata")
    if auto_retrain:
        command.append("--auto_retrain")
    return command


def preflight_live_capture(interface: str, tshark_path: str) -> tuple[bool, str, str | None]:
    try:
        resolved_tshark = find_tshark(tshark_path)
    except Exception as exc:
        return False, f"Invalid tshark path: {exc}", None

    ok, message = validate_tshark_interface(resolved_tshark, interface)
    if not ok:
        return False, message, resolved_tshark
    return True, "", resolved_tshark


def start_live_capture(command: list[str]) -> None:
    if live_running():
        st.warning("Live capture is already running.")
        return
    proc = subprocess.Popen(command, cwd=str(REPO_ROOT))
    st.session_state["live_proc"] = proc
    st.session_state["live_command"] = command
    st.session_state["live_started_at"] = pd.Timestamp.now(tz="UTC").isoformat()


def stop_live_capture() -> None:
    proc = st.session_state.get("live_proc")
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    st.session_state["live_proc"] = None


def safe_upload_name(name: str) -> str:
    cleaned = Path(name).name.strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in cleaned)


def save_uploaded_pcaps(uploaded_files, queue_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    queue_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files or []:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in {".pcap", ".pcapng", ".cap"}:
            continue
        unique_name = f"{int(time.time_ns())}_{safe_upload_name(uploaded_file.name)}"
        target = queue_dir / unique_name
        target.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target)
    return saved_paths


def _unlink_with_retry(path: Path, *, attempts: int = 3, delay_seconds: float = 0.25) -> bool:
    for attempt in range(attempts):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if attempt + 1 < attempts:
                time.sleep(delay_seconds * (attempt + 1))
                continue
            return False
        except OSError:
            return False
    return False


def clear_directory_files(folder: Path) -> dict[str, object]:
    if not folder.exists():
        return {"deleted": 0, "skipped": []}
    deleted = 0
    skipped: list[str] = []
    for path in folder.iterdir():
        if path.is_file():
            if _unlink_with_retry(path):
                deleted += 1
            else:
                skipped.append(str(path))
        elif path.is_dir():
            try:
                shutil.rmtree(path, ignore_errors=False)
            except Exception:
                skipped.append(str(path))
            else:
                deleted += 1
    return {"deleted": deleted, "skipped": skipped}


def reset_live_environment(mongo_uri: str, db_name: str) -> dict[str, object]:
    files_deleted = 0
    skipped_files: list[str] = []
    mongo_deleted: dict[str, int] = {}
    if live_running():
        stop_live_capture()
        time.sleep(3)
    try:
        db = connect_database(mongo_uri, db_name)
        mongo_deleted = clear_live_collections(db)
    except Exception as exc:
        mongo_deleted = {"error": str(exc)}

    live_root = REPO_ROOT / "output" / "live"
    for folder in [LIVE_INBOX_DIR, LIVE_PROCESSED_DIR, live_root / "captures"]:
        cleanup = clear_directory_files(folder)
        files_deleted += int(cleanup.get("deleted", 0))
        skipped_files.extend([str(path) for path in cleanup.get("skipped", [])])

    status_path = live_root / "live_capture_status.json"
    if status_path.exists():
        if _unlink_with_retry(status_path, attempts=4, delay_seconds=0.5):
            files_deleted += 1
        else:
            skipped_files.append(str(status_path))

    st.session_state["live_proc"] = None
    st.session_state["live_command"] = None
    st.session_state["live_started_at"] = None

    st.cache_data.clear()
    return {
        "mongo_deleted": mongo_deleted,
        "files_deleted": files_deleted,
        "files_skipped": len(skipped_files),
        "skipped_files": skipped_files,
        "reset_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }



@st.cache_data(show_spinner=False, ttl=5)
def load_summary_document(
    mongo_uri: str,
    db_name: str,
    kind: str,
) -> dict[str, object]:
    try:
        db = connect_database(mongo_uri, db_name)
        document = db[ANALYSIS_SUMMARIES_COLLECTION].find_one({"kind": kind}, sort=[("created_at", -1)], projection={"_id": 0})
        if document:
            if "payload" in document:
                return document["payload"]
            return document
    except Exception:
        pass
    return load_local_summary(kind)


def load_summary_text(kind: str) -> str:
    summary = load_local_summary(kind)
    return str(summary.get("text", ""))


def metric_block(label: str, value, delta: str | None = None) -> None:
    st.metric(label, value, delta=delta)


def render_overview(mongo_uri: str, db_name: str, db_connected: bool) -> None:
    st.subheader("Overview")

    summary = load_summary_document(mongo_uri, db_name, "holdout_metrics")
    retrain_summary = load_summary_document(mongo_uri, db_name, "retrain_comparison")
    anomaly_summary = load_summary_document(mongo_uri, db_name, "anomaly_holdout_metrics")
    drift_summary = load_summary_document(mongo_uri, db_name, "live_drift_summary") or load_summary_document(mongo_uri, db_name, "drift_summary")
    counts = {}
    if db_connected:
        try:
            db = connect_database(mongo_uri, db_name)
            counts = collection_counts(db)
        except Exception:
            counts = {}

    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metric_from_summary(summary, 'accuracy'):.4f}" if summary else "n/a")
    cols[1].metric("PR-AUC", f"{metric_from_summary(summary, 'pr_auc'):.4f}" if summary else "n/a")
    cols[2].metric("Recall", f"{metric_from_summary(summary, 'recall'):.4f}" if summary else "n/a")
    cols[3].metric("Attack FNR", f"{metric_from_summary(summary, 'fnr'):.4f}" if summary else "n/a")
    cols[4].metric("ROC-AUC", f"{metric_from_summary(summary, 'roc_auc'):.4f}" if summary else "n/a")

    if counts:
        st.caption(
            "MongoDB collections: "
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Retraining status**")
        if retrain_summary:
            st.caption(
                f"Triggered={retrain_summary.get('triggered')} | "
                f"Assessment={retrain_summary.get('assessment')} | "
                f"Reason={retrain_summary.get('trigger_reason')}"
            )
            with st.expander("Retraining details", expanded=False):
                st.json(
                    {
                        "triggered": retrain_summary.get("triggered"),
                        "assessment": retrain_summary.get("assessment"),
                        "trigger_reason": retrain_summary.get("trigger_reason"),
                    }
                )
        else:
            st.info("No retraining summary found.")

    with right:
        st.markdown("**Anomaly and drift status**")
        if anomaly_summary:
            st.caption(
                f"Anomaly recall={metric_from_summary(anomaly_summary, 'recall'):.4f} | "
                f"Anomaly FNR={metric_from_summary(anomaly_summary, 'fnr'):.4f}"
            )
        if drift_summary:
            st.caption(
                f"Drift severity={drift_summary.get('severity', 'n/a')} | "
                f"Max PSI={drift_summary.get('severity_stats', {}).get('max_psi', 0.0):.4f} | "
                f"Flag={drift_summary.get('drift_flag', False)}"
            )
            with st.expander("Drift details", expanded=False):
                st.json(
                    {
                        "severity": drift_summary.get("severity"),
                        "max_psi": drift_summary.get("severity_stats", {}).get("max_psi"),
                        "drift_flag": drift_summary.get("drift_flag"),
                    }
                )


def render_predictions(mongo_uri: str, db_name: str) -> None:
    st.subheader("Live Predictions")

    limit = st.slider("Rows to display", 50, 5000, 500, step=50)
    tabs = st.tabs([friendly_label(LIVE_PREDICTION_KIND), friendly_label(LIVE_ANOMALY_PREDICTION_KIND)])
    view_specs = [
        (tabs[0], LIVE_PREDICTION_KIND, "Classifier"),
        (tabs[1], LIVE_ANOMALY_PREDICTION_KIND, "Anomaly"),
    ]
    for tab, kind_label, short_name in view_specs:
        with tab:
            predictions = load_live_prediction_frame(mongo_uri, db_name, kind=kind_label, source=None, limit=limit)
            if predictions.empty:
                st.info(f"No {short_name.lower()} live prediction rows found in MongoDB yet.")
                continue

            sort_columns = [column for column in ["window_id", "record_index", "created_at"] if column in predictions.columns]
            if sort_columns:
                predictions = predictions.sort_values(sort_columns)
            display_df = predictions.head(limit).copy()

            metric_cols = st.columns(4)
            metric_cols[0].metric("Rows", len(display_df))
            metric_cols[1].metric("Windows", int(display_df["window_id"].nunique()) if "window_id" in display_df.columns else 0)
            if "pred_proba_attack" in display_df.columns:
                metric_cols[2].metric("Mean proba", f"{float(display_df['pred_proba_attack'].mean()):.4f}")
                metric_cols[3].metric("Max proba", f"{float(display_df['pred_proba_attack'].max()):.4f}")
            elif "anomaly_score" in display_df.columns:
                metric_cols[2].metric("Mean score", f"{float(display_df['anomaly_score'].mean()):.4f}")
                metric_cols[3].metric("Max score", f"{float(display_df['anomaly_score'].max()):.4f}")
            else:
                metric_cols[2].metric("Mean proba", "n/a")
                metric_cols[3].metric("Max proba", "n/a")

            score_col = "pred_proba_attack" if "pred_proba_attack" in display_df.columns else "anomaly_score" if "anomaly_score" in display_df.columns else None
            if score_col:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=display_df.get("record_index", display_df.index),
                        y=display_df[score_col],
                        mode="lines",
                        name="Attack probability" if score_col == "pred_proba_attack" else "Anomaly score",
                    )
                )
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

            table_columns = [
                column
                for column in [
                    "source",
                    "source_file",
                    "window_id",
                    "record_index",
                    "fold",
                    "true_label",
                    "true_label_name",
                    "pred_proba_attack",
                    "pred_label",
                    "pred_label_name",
                    "attack_type_name",
                    "attack_type_proba",
                    "model_version",
                    "attack_model_version",
                    "correct",
                    "anomaly_score",
                    "anomaly_pred_label",
                    "anomaly_pred_name",
                ]
                if column in display_df.columns
            ]
            with st.expander(f"{short_name} rows", expanded=False):
                st.dataframe(display_df[table_columns] if table_columns else display_df, use_container_width=True, height=340)


def render_feature_importance(mongo_uri: str, db_name: str) -> None:
    st.subheader("Live SHAP and Drift")

    tabs = st.tabs([friendly_label(LIVE_SHAP_IMPORTANCE_KIND), friendly_label(LIVE_DRIFT_IMPORTANCE_KIND)])
    view_specs = [
        (tabs[0], LIVE_SHAP_IMPORTANCE_KIND, "Shap"),
        (tabs[1], LIVE_DRIFT_IMPORTANCE_KIND, "Drift"),
    ]
    for tab, importance_type, short_name in view_specs:
        with tab:
            df = load_feature_importance_frame(mongo_uri, db_name, importance_type)
            if df.empty:
                st.info(f"No {short_name.lower()} live rows available yet.")
                continue

            if importance_type == LIVE_DRIFT_IMPORTANCE_KIND and "psi" in df.columns:
                sort_column = "psi"
            elif "normalized_importance" in df.columns:
                sort_column = "normalized_importance"
            elif "mean_abs_shap" in df.columns:
                sort_column = "mean_abs_shap"
            else:
                sort_column = df.columns[0]

            upper_bound = max(5, min(50, len(df)))
            default_top_n = min(15, upper_bound)
            top_n = st.slider(
                f"Top {short_name.lower()} features",
                5,
                upper_bound,
                default_top_n,
                key=f"{importance_type}_top_n",
            )
            top_df = df.sort_values(sort_column, ascending=False).head(top_n)
            value_col = sort_column
            if importance_type == LIVE_DRIFT_IMPORTANCE_KIND and "psi" in top_df.columns:
                value_col = "psi"

            cols = st.columns(3)
            cols[0].metric("Rows", len(top_df))
            cols[1].metric("Features", int(top_df["feature"].nunique()) if "feature" in top_df.columns else 0)
            cols[2].metric("Source", friendly_label(importance_type))

            fig = go.Figure(
                go.Bar(
                    x=top_df[value_col],
                    y=top_df["feature"],
                    orientation="h",
                    marker_color="#4c78a8" if importance_type == LIVE_SHAP_IMPORTANCE_KIND else "#f58518",
                )
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander(f"{short_name} rows", expanded=False):
                st.dataframe(top_df, use_container_width=True, height=280)


def render_alert_explanations(mongo_uri: str, db_name: str) -> None:
    st.subheader("Alert Explanations")
    window_id = st.number_input("Window id", min_value=0, max_value=1_000_000, value=0, step=1)
    limit = st.slider("Rows to display", 10, 500, 100, step=10)
    df = load_live_alert_frame(mongo_uri, db_name, window_id=int(window_id) if window_id > 0 else None, limit=limit)
    if df.empty:
        st.info("No live alert explanation rows available yet.")
        return
    cols = st.columns(4)
    cols[0].metric("Rows", len(df))
    cols[1].metric("Windows", int(df["window_id"].nunique()) if "window_id" in df.columns else 0)
    cols[2].metric("Pred attack", int((df.get("pred_label", 0) == 1).sum()) if "pred_label" in df.columns else 0)
    cols[3].metric("Model versions", int(df["model_version"].nunique()) if "model_version" in df.columns else 0)
    display_cols = [
        column
        for column in [
            "window_id",
            "record_index",
            "prediction_id",
            "pred_label_name",
            "pred_proba_attack",
            "attack_type_name",
            "attack_type_proba",
            "model_version",
            "attack_model_version",
            "top_features",
        ]
        if column in df.columns
    ]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, height=360)


def render_drift_and_retraining(mongo_uri: str, db_name: str) -> None:
    st.subheader("Drift and Retraining")
    left, right = st.columns(2)
    with left:
        drift_df = load_drift_event_frame(mongo_uri, db_name)
        if drift_df.empty:
            st.info("No ADWIN drift events recorded yet.")
        else:
            st.metric("Drift events", len(drift_df))
            st.metric("Latest confidence", f"{float(drift_df['confidence'].iloc[0]):.4f}" if "confidence" in drift_df.columns else "n/a")
            st.dataframe(
                drift_df[[col for col in ["created_at", "window_id", "record_index", "confidence", "drift_count", "running_mean", "running_std"] if col in drift_df.columns]],
                use_container_width=True,
                height=260,
            )
    with right:
        retrain_df = load_retrain_event_frame(mongo_uri, db_name)
        if retrain_df.empty:
            st.info("No retraining events recorded yet.")
        else:
            st.metric("Retrain events", len(retrain_df))
            st.metric("Latest result", str(retrain_df["kind"].iloc[0]) if "kind" in retrain_df.columns else "n/a")
            st.dataframe(
                retrain_df[[col for col in ["created_at", "kind", "window_id", "reason", "exit_code", "accepted", "assessment"] if col in retrain_df.columns]],
                use_container_width=True,
                height=260,
            )


def render_robustness(mongo_uri: str, db_name: str) -> None:
    st.subheader("Robustness")
    df = load_adversarial_frame(mongo_uri, db_name)
    if df.empty:
        st.info("No adversarial robustness report found yet.")
        return
    cols = st.columns(4)
    record = df.iloc[0].to_dict()
    payload = record.get("payload", record) or {}
    clean = payload.get("clean_metrics", {}) or {}
    fgsm = payload.get("fgsm_metrics", {}) or {}
    pgd = payload.get("pgd_metrics", {}) or {}
    cols[0].metric("Rows", len(df))
    cols[1].metric("Clean recall", f"{float(clean.get('attack_recall', 0.0)):.4f}")
    cols[2].metric("FGSM recall", f"{float(fgsm.get('attack_recall', 0.0)):.4f}")
    cols[3].metric("PGD recall", f"{float(pgd.get('attack_recall', 0.0)):.4f}")
    st.json(payload)


def render_thresholds(mongo_uri: str, db_name: str) -> None:
    st.subheader("Threshold Calibration")

    source_label_map = [("Live", "live"), ("Holdout", "holdout"), ("CV", "cv")]
    source_label = st.selectbox("Threshold source", [label for label, _ in source_label_map], index=0)
    source = dict(source_label_map)[source_label]
    df = load_threshold_frame(mongo_uri, db_name, source)
    if df.empty:
        if source == "live":
            st.info("No live threshold rows available yet. They will appear once labeled live windows are available in MongoDB.")
        else:
            st.warning("No threshold calibration rows available.")
        return

    metric_cols = [col for col in ["attack_precision", "attack_recall", "attack_fnr", "attack_f1", "attack_f2"] if col in df.columns]
    threshold_grid = df.sort_values("threshold")

    fig = go.Figure()
    for column in metric_cols:
        fig.add_trace(go.Scatter(x=threshold_grid["threshold"], y=threshold_grid[column], mode="lines", name=column))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    if source == "live":
        st.caption("Source: Live (derived from labeled live windows in MongoDB)")
    else:
        st.caption(f"Source: {source_label}")
    recommended_rows = threshold_grid.loc[threshold_grid["threshold"].isin([0.05, 0.06, 0.13, 0.23])]
    if source == "live":
        try:
            recommendations = select_recommendations(threshold_grid, min_precision=0.97, default_threshold=0.5)
            rec = recommendations.get("recommended", {})
            sample_count = int(threshold_grid["sample_count"].iloc[0]) if "sample_count" in threshold_grid.columns and not threshold_grid.empty else len(threshold_grid)
            cols = st.columns(4)
            cols[0].metric("Max F1", f"{rec['max_f1']['threshold']:.2f}")
            cols[1].metric("Max F2", f"{rec['max_f2']['threshold']:.2f}")
            cols[2].metric("Best recall", f"{rec['highest_recall_under_min_precision']['threshold']:.2f}")
            cols[3].metric("Lowest FNR", f"{rec['lowest_fnr_under_min_precision']['threshold']:.2f}")
            with st.expander("Recommendations", expanded=False):
                st.text(
                    "\n".join(
                        [
                            f"Samples: {sample_count}",
                            f"Max F1 threshold: {rec['max_f1']['threshold']:.2f}",
                            f"Max F2 threshold: {rec['max_f2']['threshold']:.2f}",
                            f"Best recall under precision constraint: {rec['highest_recall_under_min_precision']['threshold']:.2f}",
                            f"Lowest FNR under precision constraint: {rec['lowest_fnr_under_min_precision']['threshold']:.2f}",
                        ]
                    )
                )
        except Exception:
            pass
    else:
        recommendation_docs = load_summary_document(mongo_uri, db_name, "threshold_calibration_summary")
        if recommendation_docs:
            with st.expander("Recommendations", expanded=False):
                st.text(recommendation_docs.get("text", "") if isinstance(recommendation_docs, dict) else str(recommendation_docs))
    with st.expander("Threshold rows", expanded=False):
        st.dataframe(recommended_rows if not recommended_rows.empty else threshold_grid.head(20), use_container_width=True, height=260)


def render_live_capture(mongo_uri: str, db_name: str) -> None:
    st.subheader("Live Capture")

    reset_notice = st.session_state.pop("live_reset_notice", None)
    if reset_notice:
        st.success(reset_notice)
        skipped_files = st.session_state.pop("live_reset_skipped_files", None)
        if skipped_files:
            with st.expander("Skipped locked files", expanded=False):
                st.write(skipped_files)

    left, right = st.columns(2)
    with left:
        interface = st.text_input("Interface / source", value=st.session_state.get("live_interface", ""))
        tshark_path = st.text_input("tshark path", value=st.session_state.get("live_tshark", "tshark"))
        model_path = st.text_input(
            "Active model pointer",
            value=st.session_state.get("live_model_path", str(resolve_live_model_reference())),
        )
        window_seconds = st.slider("Capture window seconds", 5, 120, int(st.session_state.get("live_window_seconds", 30)), step=5)
        packet_count = st.number_input("Packet count per window (0 = disabled)", min_value=0, max_value=50000, value=int(st.session_state.get("live_packet_count", 0)), step=10)
        capture_filter = st.text_input("Capture filter", value=st.session_state.get("live_capture_filter", ""))
        display_filter = st.text_input("Display filter", value=st.session_state.get("live_display_filter", ""))
        refresh_ms = st.slider("Dashboard refresh ms", 1000, 10000, int(st.session_state.get("live_refresh_ms", 3000)), step=500)
    with right:
        threshold = st.slider("Live decision threshold", 0.0, 1.0, float(st.session_state.get("live_threshold", 0.5)), step=0.01)
        file_max_threshold = st.slider("File max threshold", 0.0, 1.0, float(st.session_state.get("live_file_max_threshold", 0.5)), step=0.01)
        file_ratio_threshold = st.slider("File ratio threshold", 0.0, 1.0, float(st.session_state.get("live_file_ratio_threshold", 0.4)), step=0.01)
        min_records = st.number_input("Minimum records", min_value=1, max_value=5000, value=int(st.session_state.get("live_min_records", 1)), step=1)
        shap_sample_rows = st.number_input("SHAP sample rows", min_value=1, max_value=1000, value=int(st.session_state.get("live_shap_sample_rows", 50)), step=5)
        shap_top_n = st.number_input("SHAP top features", min_value=1, max_value=50, value=int(st.session_state.get("live_shap_top_n", 15)), step=1)
        pause_seconds = st.number_input("Pause between windows", min_value=0.0, max_value=60.0, value=float(st.session_state.get("live_pause_seconds", 0.0)), step=0.5)
        max_windows = st.number_input("Max windows (0 = continuous)", min_value=0, max_value=100000, value=int(st.session_state.get("live_max_windows", 0)), step=1)
        auto_retrain = st.checkbox("Auto retrain on drift", value=bool(st.session_state.get("live_auto_retrain", False)))

    st.session_state["live_interface"] = interface
    st.session_state["live_tshark"] = tshark_path
    st.session_state["live_model_path"] = model_path
    st.session_state["live_window_seconds"] = window_seconds
    st.session_state["live_packet_count"] = int(packet_count)
    st.session_state["live_capture_filter"] = capture_filter
    st.session_state["live_display_filter"] = display_filter
    st.session_state["live_refresh_ms"] = refresh_ms
    st.session_state["live_threshold"] = threshold
    st.session_state["live_file_max_threshold"] = file_max_threshold
    st.session_state["live_file_ratio_threshold"] = file_ratio_threshold
    st.session_state["live_min_records"] = int(min_records)
    st.session_state["live_shap_sample_rows"] = int(shap_sample_rows)
    st.session_state["live_shap_top_n"] = int(shap_top_n)
    st.session_state["live_pause_seconds"] = float(pause_seconds)
    st.session_state["live_max_windows"] = int(max_windows)
    st.session_state["live_auto_retrain"] = bool(auto_retrain)
    start_error = st.session_state.pop("live_start_error", None)
    if start_error:
        st.error(start_error)

    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("Start Capture", type="primary", use_container_width=True):
            if not interface.strip():
                st.error("Enter a capture interface or adapter name before starting live capture.")
            elif live_running():
                st.warning("Live capture is already running.")
            else:
                ok, message, resolved_tshark = preflight_live_capture(interface.strip(), tshark_path.strip())
                if not ok or resolved_tshark is None:
                    st.session_state["live_start_error"] = message
                    st.error(message)
                else:
                    reset_result = reset_live_environment(mongo_uri, db_name)
                    st.cache_data.clear()
                    command = build_live_command(
                        mongo_uri=mongo_uri,
                        db_name=db_name,
                        interface=interface.strip(),
                        tshark=resolved_tshark,
                        model_path=model_path.strip(),
                        model_pointer_path=model_path.strip(),
                        window_seconds=window_seconds,
                        packet_count=int(packet_count) if int(packet_count) > 0 else None,
                        capture_filter=capture_filter.strip() or None,
                        display_filter=display_filter.strip() or None,
                        threshold=threshold,
                        file_max_threshold=file_max_threshold,
                        file_ratio_threshold=file_ratio_threshold,
                        min_records=int(min_records),
                        shap_sample_rows=int(shap_sample_rows),
                        shap_top_n=int(shap_top_n),
                        no_metadata=False,
                        max_windows=int(max_windows),
                        pause_seconds=float(pause_seconds),
                        auto_retrain=auto_retrain,
                        output_dir=str(REPO_ROOT / "output" / "live"),
                        capture_dir=str(REPO_ROOT / "output" / "live" / "captures"),
                        injection_dir=str(LIVE_INBOX_DIR),
                        processed_dir=str(LIVE_PROCESSED_DIR),
                        status_path=str(REPO_ROOT / "output" / "live" / "live_capture_status.json"),
                    )
                    try:
                        start_live_capture(command)
                        skipped_files = reset_result.get("skipped_files", [])
                        msg = (
                            "Fresh live session started."
                            + f" Cleared Mongo docs={sum(reset_result['mongo_deleted'].values())},"
                            + f" files={reset_result['files_deleted']}."
                        )
                        if skipped_files:
                            msg += f" Skipped locked files={len(skipped_files)}."
                        st.success(msg)
                        if skipped_files:
                            with st.expander("Skipped files", expanded=False):
                                st.write(skipped_files)
                    except Exception as exc:
                        st.error(f"Failed to start live capture: {exc}")
    with action_cols[1]:
        if st.button("Stop Capture", use_container_width=True):
            stop_live_capture()
            st.warning("Live capture stop requested.")
    with action_cols[2]:
        st.metric("Live status", live_status_text())
    with action_cols[3]:
        if st.button("Reset Live Data", use_container_width=True):
            result = reset_live_environment(mongo_uri, db_name)
            st.cache_data.clear()
            st.session_state["live_reset_notice"] = (
                "Live data reset completed. "
                f"Mongo cleared: {result['mongo_deleted']}, files deleted: {result['files_deleted']}"
            )
            if result.get("files_skipped", 0):
                st.session_state["live_reset_notice"] += f", skipped: {result['files_skipped']}"
                st.session_state["live_reset_skipped_files"] = result.get("skipped_files", [])
            st.rerun()

    st.markdown("**PCAP injection**")
    with st.form("live_pcap_injection_form", clear_on_submit=False):
        uploaded_pcaps = st.file_uploader(
            "Upload .pcap files to inject into the running live stream",
            type=["pcap", "pcapng", "cap"],
            accept_multiple_files=True,
            help="Uploaded PCAPs are queued to the live worker inbox and processed as injected live windows.",
        )
        inject_submitted = st.form_submit_button("Inject Uploaded PCAPs")
        if inject_submitted:
            if not live_running():
                st.warning("Start live capture first, then inject PCAPs into the running worker.")
            elif not uploaded_pcaps:
                st.warning("Choose one or more PCAP files before injecting.")
            else:
                saved_paths = save_uploaded_pcaps(uploaded_pcaps, LIVE_INBOX_DIR)
                if saved_paths:
                    st.success(f"Queued {len(saved_paths)} PCAP file(s) for injection.")
                    with st.expander("Queued files", expanded=False):
                        st.write([str(path) for path in saved_paths])
                else:
                    st.warning("No valid PCAP files were uploaded.")

    if live_running():
        st_autorefresh(interval=int(refresh_ms), key="live_capture_refresh")

    live_status = load_live_status()
    live_window_count = load_live_window_count(mongo_uri, db_name, source=None)
    live_windows = load_live_window_frame(mongo_uri, db_name, source=None, limit=25)
    latest_window_id = int(live_windows.iloc[0]["window_id"]) if not live_windows.empty and "window_id" in live_windows.columns else None
    live_threshold_rows = load_threshold_frame(mongo_uri, db_name, "live")
    live_placeholder = st.empty()
    with live_placeholder.container():
        latest = live_windows.iloc[0].to_dict() if not live_windows.empty else {}
        cols = st.columns(4)
        cols[0].metric("Window count", int(live_window_count))
        cols[1].metric("Latest window", int(latest.get("window_id", 0)) if latest_window_id is not None else "n/a")
        cols[2].metric("Records", int(latest.get("records", 0)))
        cols[3].metric("Threshold rows", int(len(live_threshold_rows)))

        health_cols = st.columns(4)
        health_cols[0].metric("Mean proba", f"{float(latest.get('mean_attack_probability', 0.0)):.4f}")
        health_cols[1].metric("Max proba", f"{float(latest.get('max_attack_probability', 0.0)):.4f}")
        health_cols[2].metric("Window label", int(latest.get("window_pred_label", 0)))
        health_cols[3].metric("Worker", "error" if live_status.get("last_error") else "ok")

        if live_status.get("last_error"):
            st.error(f"Live worker error: {live_status.get('last_error')}")

        if live_windows.empty:
            st.info("No live windows available yet. Start capture to populate MongoDB.")
        else:
            st.caption("Live windows are counted directly from MongoDB and remain visible across reruns.")

            live_predictions = load_live_prediction_frame(mongo_uri, db_name, kind=LIVE_PREDICTION_KIND, source=None, window_id=latest_window_id, limit=1000)
            if live_predictions.empty:
                st.warning("No live classifier prediction rows found for the latest window.")
            else:
                thresholded = live_predictions.copy()
                thresholded["dynamic_pred_label"] = (thresholded["pred_proba_attack"] >= threshold).astype(int)

                if "true_label" in thresholded.columns and thresholded["true_label"].notna().any():
                    labeled = thresholded[thresholded["true_label"].notna()].copy()
                    metrics = evaluation_from_predictions(labeled["true_label"].astype(int), labeled["pred_proba_attack"].to_numpy(), threshold=threshold)
                    metric_cols = st.columns(5)
                    metric_cols[0].metric("Accuracy", f"{metrics['metrics']['accuracy']:.4f}")
                    metric_cols[1].metric("Precision", f"{metrics['metrics']['precision']:.4f}")
                    metric_cols[2].metric("Recall", f"{metrics['metrics']['recall']:.4f}")
                    metric_cols[3].metric("FNR", f"{metrics['metrics']['fnr']:.4f}")
                    metric_cols[4].metric("ROC-AUC", f"{metrics['metrics']['roc_auc']:.4f}")
                else:
                    metric_cols = st.columns(5)
                    metric_cols[0].metric("Accuracy", "n/a")
                    metric_cols[1].metric("Precision", "n/a")
                    metric_cols[2].metric("Recall", "n/a")
                    metric_cols[3].metric("FNR", "n/a")
                    metric_cols[4].metric("ROC-AUC", "n/a")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=thresholded["record_index"], y=thresholded["pred_proba_attack"], mode="lines", name="Attack probability"))
                fig.add_hline(y=threshold, line_dash="dash", line_color="#d62728", annotation_text=f"threshold={threshold:.2f}")
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

                if "dynamic_pred_label" in thresholded.columns:
                    live_stats = st.columns(3)
                    live_stats[0].metric("Attack rate", f"{float(thresholded['dynamic_pred_label'].mean()):.4f}")
                    live_stats[1].metric("Threshold", f"{threshold:.2f}")
                    live_stats[2].metric("Rows", int(len(thresholded)))

                display_cols = [column for column in ["record_index", "pred_proba_attack", "pred_label", "dynamic_pred_label", "true_label", "pred_label_name", "true_label_name"] if column in thresholded.columns]
                with st.expander("Live prediction rows", expanded=False):
                    st.dataframe(thresholded[display_cols], use_container_width=True, height=260)

            live_shap = load_live_shap_frame(mongo_uri, db_name, source=None, window_id=latest_window_id, limit=shap_top_n)
            if not live_shap.empty and "feature" in live_shap.columns:
                shap_sort_col = "normalized_importance" if "normalized_importance" in live_shap.columns else "mean_abs_contrib"
                live_shap = live_shap.sort_values(shap_sort_col, ascending=False).head(shap_top_n)
                fig = go.Figure(go.Bar(x=live_shap[shap_sort_col], y=live_shap["feature"], orientation="h", marker_color="#54a24b"))
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Live SHAP rows", expanded=False):
                    st.dataframe(live_shap, use_container_width=True, height=260)
            else:
                st.info("No live SHAP rows available yet.")

            with st.expander("Live window summaries", expanded=False):
                st.dataframe(live_windows, use_container_width=True, height=280)


def render_analysis_tabs(mongo_uri: str, db_name: str) -> None:
    st.subheader("Anomaly, Drift, and Retraining")
    tabs = st.tabs(["Anomaly", "Drift", "Retraining"])

    with tabs[0]:
        anomaly = load_summary_document(mongo_uri, db_name, "anomaly_holdout_metrics")
        if anomaly:
            cols = st.columns(4)
            cols[0].metric("Attack precision", f"{metric_from_summary(anomaly, 'precision'):.4f}")
            cols[1].metric("Attack recall", f"{metric_from_summary(anomaly, 'recall'):.4f}")
            cols[2].metric("Attack FNR", f"{metric_from_summary(anomaly, 'fnr'):.4f}")
            cols[3].metric("ROC-AUC", f"{metric_from_summary(anomaly, 'roc_auc'):.4f}")
            with st.expander("Anomaly details", expanded=False):
                st.json(anomaly)
        else:
            st.info("No anomaly summary available.")

    with tabs[1]:
        drift = load_summary_document(mongo_uri, db_name, "live_drift_summary") or load_summary_document(mongo_uri, db_name, "drift_summary")
        if drift:
            cols = st.columns(3)
            cols[0].metric("Severity", drift.get("severity", "n/a"))
            cols[1].metric("Max PSI", f"{drift.get('severity_stats', {}).get('max_psi', 0.0):.4f}")
            cols[2].metric("Drift flag", str(drift.get("drift_flag", False)))
            top_features = drift.get("top_features", [])
            if top_features:
                drift_df = pd.DataFrame(top_features)
                value_col = "psi" if "psi" in drift_df.columns else drift_df.columns[-1]
                fig = go.Figure(
                    go.Bar(
                        x=drift_df[value_col],
                        y=drift_df["feature"],
                        orientation="h",
                        marker_color="#f58518",
                    )
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Drift details", expanded=False):
                    st.dataframe(drift_df, use_container_width=True, height=280)
        else:
            st.info("No drift summary available.")

    with tabs[2]:
        retrain = load_summary_document(mongo_uri, db_name, "retrain_comparison")
        trigger = load_summary_document(mongo_uri, db_name, "retrain_trigger")
        if retrain:
            st.markdown("**Retraining trigger**")
            with st.expander("Trigger details", expanded=False):
                st.json(trigger)
            comparison = retrain.get("comparison", [])
            if comparison:
                comp_df = pd.DataFrame(comparison)
                st.dataframe(comp_df, use_container_width=True, height=260)
                if {"metric", "original", "retrained"}.issubset(comp_df.columns):
                    melted = comp_df.melt(id_vars="metric", value_vars=["original", "retrained"], var_name="model", value_name="value")
                    fig = go.Figure()
                    for model_name in melted["model"].unique():
                        model_df = melted[melted["model"] == model_name]
                        fig.add_trace(go.Bar(x=model_df["metric"], y=model_df["value"], name=model_name))
                    fig.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No retraining summary available.")


def render_connection_banner(mongo_uri: str, db_name: str) -> bool:
    try:
        db = connect_database(mongo_uri, db_name)
        st.sidebar.success("MongoDB connected")
        st.sidebar.caption(f"{db_name} @ {mongo_uri}")
        counts = collection_counts(db)
        st.sidebar.markdown("**Collection counts**")
        for name, count in counts.items():
            st.sidebar.write(f"{name}: {count}")
        return True
    except Exception as exc:
        st.sidebar.warning("MongoDB unavailable, using local artifacts.")
        st.sidebar.caption(str(exc))
        return False


def main() -> None:
    st.set_page_config(page_title="BigDataFinalPaper Dashboard", layout="wide")
    st.title("BigDataFinalPaper Dashboard")
    st.caption("MongoDB-backed view of the offline Edge-IIoT pipeline.")

    mongo_uri = st.sidebar.text_input("MongoDB URI", value=DEFAULT_MONGO_URI)
    db_name = st.sidebar.text_input("Database name", value=DEFAULT_MONGO_DB)
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    db_connected = render_connection_banner(mongo_uri, db_name)

    render_live_capture(mongo_uri, db_name)
    st.divider()
    render_overview(mongo_uri, db_name, db_connected)
    st.divider()
    render_predictions(mongo_uri, db_name)
    st.divider()
    render_feature_importance(mongo_uri, db_name)
    st.divider()
    render_thresholds(mongo_uri, db_name)
    st.divider()
    render_analysis_tabs(mongo_uri, db_name)
    st.divider()
    render_alert_explanations(mongo_uri, db_name)
    st.divider()
    render_drift_and_retraining(mongo_uri, db_name)
    st.divider()
    render_robustness(mongo_uri, db_name)

    with st.expander("MongoDB setup notes"):
        st.write(
            {
                "uri": mongo_uri,
                "database": db_name,
                "raw_packets": "seeded from demo extracted CSVs in this repo; live packet documents can be added later",
                "collections": [
                    "raw_packets",
                    "feature_vectors",
                    "predictions",
                    "alert_explanations",
                    "live_windows",
                    "feature_importance",
                    "threshold_metrics",
                    "drift_events",
                    "retrain_events",
                    "adversarial_evaluations",
                    "analysis_summaries",
                ],
            }
        )


if __name__ == "__main__":
    main()
