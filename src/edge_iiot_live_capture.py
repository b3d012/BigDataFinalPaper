from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import threading
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from edge_iiot_demo_replay import (
    available_tshark_fields,
    available_tshark_interfaces,
    bundle_field_contract,
    find_tshark,
    prepare_model_input,
    resolve_supported_fields,
    validate_tshark_interface,
)
from edge_iiot_adwin import ADWINMonitor
from edge_iiot_thresholds import build_threshold_grid, compute_threshold_metrics, select_recommendations
from edge_iiot_experiment import DEFAULT_EDGE_CSV, get_transformed_feature_names
from edge_iiot_runtime import DEFAULT_ACTIVE_MODEL_POINTER_PATH, load_active_model_pointer, json_safe, resolve_model_path
from edge_iiot_drift import compare_batches, load_dataset_batches
from edge_iiot_mongo import (
    ANALYSIS_SUMMARIES_COLLECTION,
    ALERT_EXPLANATIONS_COLLECTION,
    ADVERSARIAL_EVALUATIONS_COLLECTION,
    DRIFT_EVENTS_COLLECTION,
    FEATURE_IMPORTANCE_COLLECTION,
    LIVE_WINDOWS_COLLECTION,
    RETRAIN_EVENTS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RAW_PACKETS_COLLECTION,
    FEATURE_VECTORS_COLLECTION,
    THRESHOLD_METRICS_COLLECTION,
    connect_database,
    collection_to_dataframe,
    dataframe_to_documents,
    ensure_indexes,
    insert_documents,
    upsert_documents,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "edge_iiot_xgb_model.joblib"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "live"
DEFAULT_CAPTURE_DIR = DEFAULT_OUTPUT_DIR / "captures"
DEFAULT_INJECTION_DIR = DEFAULT_OUTPUT_DIR / "inbox"
DEFAULT_PROCESSED_DIR = DEFAULT_OUTPUT_DIR / "processed"
DEFAULT_STATUS_PATH = DEFAULT_OUTPUT_DIR / "live_capture_status.json"
DEFAULT_MODEL_POINTER_PATH = DEFAULT_ACTIVE_MODEL_POINTER_PATH
DEFAULT_ANOMALY_MODEL_PATH = REPO_ROOT / "models" / "edge_iiot_isolation_forest.joblib"
DEFAULT_DRIFT_N_BINS = 10
DEFAULT_DRIFT_MIN_CATEGORY_COUNT = 5
DEFAULT_ADWIN_DELTA = 0.002
DEFAULT_ADWIN_MIN_WINDOW = 20
DEFAULT_ADWIN_MAX_WINDOW = 4096
DEFAULT_METADATA_FIELDS = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "tcp.flags",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.connection.rst",
    "tcp.connection.fin", 
    "http.request.method",
    "dns.qry.name",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def live_window_tag(window_id: int) -> str:
    return f"live_window_{window_id:06d}"


def convert_datetime(obj):
    """Convert datetime objects to JSON-safe ISO 8601 strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {key: convert_datetime(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_datetime(item) for item in obj]
    return obj


def write_status(file_path, payload):
    payload = convert_datetime(payload)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def read_status(file_path: Path) -> dict[str, object]:
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_live_status(
    *,
    status_path: Path,
    base_status: dict[str, object],
    last_window_id: int,
    last_summary: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
    state: str = "running",
) -> None:
    payload: dict[str, object] = {
        **base_status,
        "state": state,
        "last_window_id": int(last_window_id),
        "updated_at": utc_now().isoformat(),
    }
    if last_summary is not None:
        payload["last_summary"] = last_summary
    if extra:
        payload.update(extra)
    write_status(status_path, payload)


def write_live_status_success(
    *,
    status_path: Path,
    base_status: dict[str, object],
    last_window_id: int,
    last_summary: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
    state: str = "running",
) -> None:
    merged_extra = dict(extra or {})
    merged_extra["last_error"] = None
    write_live_status(
        status_path=status_path,
        base_status=base_status,
        last_window_id=last_window_id,
        last_summary=last_summary,
        extra=merged_extra,
        state=state,
    )


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


def infer_attack_label_from_name(name: str | None) -> str | None:
    if not name:
        return None
    text = str(name).strip()
    if not text:
        return None
    lowered = text.lower()
    if any(token in lowered for token in ("normal", "benign", "noattack", "no_attack", "baseline")):
        return "Normal"
    return Path(text).stem.replace("_", " ").replace("-", " ").strip() or None


def validate_capture_interface(tshark: str, interface: str) -> tuple[bool, str]:
    return validate_tshark_interface(tshark, interface)


def load_bundle(model_path: Path) -> dict[str, object]:
    resolved_path = model_path
    if not resolved_path.exists():
        if resolved_path.suffix.lower() == ".json":
            fallback_joblib = resolved_path.with_name("edge_iiot_xgb_model.joblib")
            if fallback_joblib.exists():
                resolved_path = fallback_joblib
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {model_path}")
    if resolved_path.suffix.lower() == ".json":
        pointer = load_active_model_pointer(resolved_path)
        if not pointer:
            fallback_joblib = resolved_path.with_name("edge_iiot_xgb_model.joblib")
            if fallback_joblib.exists():
                resolved_path = fallback_joblib
            else:
                raise ValueError(f"Model pointer is missing or invalid: {resolved_path}")
        if resolved_path.suffix.lower() != ".json":
            return load_bundle(resolved_path)
        binary_bundle = joblib.load(resolve_model_path(resolved_path, "binary_model_path"))
        required = {"model", "preprocessor", "threshold", "training_meta"}
        missing = required - set(binary_bundle.keys())
        if missing:
            raise ValueError(f"Binary model bundle is missing required keys: {sorted(missing)}")
        binary_bundle = dict(binary_bundle)
        binary_bundle["model_version"] = pointer.get("version")
        binary_bundle["active_pointer"] = pointer
        attack_model_path = pointer.get("attack_model_path")
        if attack_model_path:
            attack_bundle = joblib.load(Path(attack_model_path))
            if {"model", "preprocessor", "classes_"}.difference(attack_bundle.keys()):
                raise ValueError(f"Attack model bundle is missing required keys: {attack_model_path}")
            binary_bundle["attack_bundle"] = attack_bundle
            binary_bundle["attack_model_version"] = attack_bundle.get("model_version")
        return binary_bundle

    bundle = joblib.load(resolved_path)
    required = {"model", "preprocessor", "threshold", "training_meta"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {sorted(missing)}")
    return bundle


def load_anomaly_bundle(model_path: Path) -> dict[str, object] | None:
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    required = {"model", "preprocessor", "threshold", "training_meta"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Anomaly bundle is missing required keys: {sorted(missing)}")
    return bundle


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df.loc[:, ~df.columns.duplicated()]


def run_capture_window_to_csv(
    *,
    tshark: str,
    interface: str,
    duration_seconds: int,
    output_csv: Path,
    fields: list[str],
    valid_fields: set[str] | None,
    capture_filter: str | None,
    display_filter: str | None,
    packet_count: int | None,
    timeout_seconds: int | None = None,
) -> tuple[list[str], list[str]]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    supported_fields, unsupported_fields = resolve_supported_fields(fields, valid_fields)

    command = [
        tshark,
        "-i",
        str(interface),
        "-a",
        f"duration:{duration_seconds}",
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        "-E",
        "occurrence=f",
    ]
    if packet_count is not None and packet_count > 0:
        command.extend(["-c", str(packet_count)])
    if capture_filter:
        command.extend(["-f", capture_filter])
    if display_filter:
        command.extend(["-Y", display_filter])

    for field in supported_fields:
        command.extend(["-e", field])

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        result = subprocess.run(command, stdout=fh, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "tshark capture failed")

    if output_csv.exists() and output_csv.stat().st_size > 0:
        df = pd.read_csv(output_csv, low_memory=False)
        df = normalize_columns(df)
    else:
        df = pd.DataFrame(columns=supported_fields)

    for field in fields:
        if field not in df.columns:
            df[field] = ""
    df = df[fields]
    df.to_csv(output_csv, index=False)
    return supported_fields, unsupported_fields


def run_pcap_to_csv(
    *,
    tshark: str,
    input_pcap: Path,
    output_csv: Path,
    fields: list[str],
    valid_fields: set[str] | None,
    display_filter: str | None,
    packet_count: int | None,
    timeout_seconds: int | None = None,
) -> tuple[list[str], list[str]]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    supported_fields, unsupported_fields = resolve_supported_fields(fields, valid_fields)

    command = [
        tshark,
        "-r",
        str(input_pcap),
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        "-E",
        "occurrence=f",
    ]
    if packet_count is not None and packet_count > 0:
        command.extend(["-c", str(packet_count)])
    if display_filter:
        command.extend(["-Y", display_filter])
    for field in supported_fields:
        command.extend(["-e", field])

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        result = subprocess.run(command, stdout=fh, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "tshark pcap extraction failed")

    if output_csv.exists() and output_csv.stat().st_size > 0:
        df = pd.read_csv(output_csv, low_memory=False)
        df = normalize_columns(df)
    else:
        df = pd.DataFrame(columns=supported_fields)

    for field in fields:
        if field not in df.columns:
            df[field] = ""
    df = df[fields]
    df.to_csv(output_csv, index=False)
    return supported_fields, unsupported_fields


def injection_pcap_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            *folder.glob("*.pcap"),
            *folder.glob("*.pcapng"),
            *folder.glob("*.cap"),
        ],
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
    )


def archive_injected_pcap(pcap_path: Path, processed_dir: Path, *, window_id: int) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{live_window_tag(window_id)}_{pcap_path.name}"
    destination = processed_dir / safe_name
    return pcap_path.replace(destination)


def prepare_window_frame(df: pd.DataFrame, bundle: dict[str, object]) -> pd.DataFrame:
    return prepare_model_input(df, bundle)


def safe_mean_absolute_contrib(transformed, model, feature_names: list[str], sample_rows: int | None = None) -> pd.DataFrame:
    if transformed is None or transformed.shape[0] == 0:
        return pd.DataFrame(columns=["feature", "mean_abs_contrib", "mean_contrib", "normalized_importance"])

    if sample_rows is not None and transformed.shape[0] > sample_rows:
        transformed = transformed[:sample_rows]

    if hasattr(transformed, "toarray"):
        matrix = transformed.toarray()
    else:
        matrix = np.asarray(transformed)

    dmat = xgb.DMatrix(matrix, feature_names=feature_names)
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    if contribs.ndim != 2 or contribs.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "mean_abs_contrib", "mean_contrib", "normalized_importance"])

    contribs = contribs[:, :-1]
    mean_abs = np.abs(contribs).mean(axis=0)
    mean_contrib = contribs.mean(axis=0)
    total = float(mean_abs.sum()) if float(mean_abs.sum()) > 0 else 1.0
    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_contrib": mean_abs,
            "mean_contrib": mean_contrib,
            "normalized_importance": mean_abs / total,
        }
    ).sort_values("mean_abs_contrib", ascending=False)


def safe_alert_contribs(
    transformed,
    model,
    feature_names: list[str],
    *,
    sample_rows: int | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    if transformed is None or transformed.shape[0] == 0:
        return pd.DataFrame(columns=["record_index", "feature", "contribution", "abs_contribution", "rank", "base_value", "expected_value"])

    if sample_rows is not None and transformed.shape[0] > sample_rows:
        transformed = transformed[:sample_rows]

    if hasattr(transformed, "toarray"):
        matrix = transformed.toarray()
    else:
        matrix = np.asarray(transformed)

    dmat = xgb.DMatrix(matrix, feature_names=feature_names)
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    if contribs.ndim != 2 or contribs.shape[1] < 2:
        return pd.DataFrame(columns=["record_index", "feature", "contribution", "abs_contribution", "rank", "base_value", "expected_value"])

    bias = contribs[:, -1]
    contribs = contribs[:, :-1]
    rows: list[dict[str, object]] = []
    for record_index, row in enumerate(contribs):
        order = np.argsort(np.abs(row))[::-1][:top_n]
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "record_index": int(record_index),
                    "feature": feature_names[int(feature_index)],
                    "contribution": float(row[int(feature_index)]),
                    "abs_contribution": float(abs(row[int(feature_index)])),
                    "rank": int(rank),
                    "base_value": float(bias[int(record_index)]),
                    "expected_value": float(bias[int(record_index)]),
                }
            )
    return pd.DataFrame(rows)


def score_model_frame(
    *,
    raw_df: pd.DataFrame,
    bundle: dict[str, object],
    threshold: float,
    attack_threshold: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, pd.DataFrame, pd.DataFrame]:
    X = prepare_window_frame(raw_df, bundle)
    transformed = bundle["preprocessor"].transform(X)
    model = bundle["model"]
    proba = model.predict_proba(transformed)[:, 1]
    pred = (proba >= threshold).astype(int)

    attack_proba: np.ndarray | None = None
    attack_pred: np.ndarray | None = None
    attack_bundle = bundle.get("attack_bundle")
    if attack_bundle is not None:
        attack_transformed = attack_bundle["preprocessor"].transform(X)
        attack_proba_matrix = attack_bundle["model"].predict_proba(attack_transformed)
        attack_pred = np.argmax(attack_proba_matrix, axis=1)
        attack_proba = attack_proba_matrix.max(axis=1)

    feature_names = get_transformed_feature_names(bundle["preprocessor"])
    shap_df = safe_mean_absolute_contrib(
        transformed,
        bundle["model"],
        feature_names,
    )
    alert_df = safe_alert_contribs(
        transformed,
        bundle["model"],
        feature_names,
        top_n=10,
    )
    return X, proba, pred, attack_proba, attack_pred, shap_df, alert_df


def score_anomaly_window(
    *,
    raw_df: pd.DataFrame,
    raw_csv: Path,
    anomaly_bundle: dict[str, object],
    threshold: float,
    interface: str,
    source_file_tag: str,
    window_id: int,
    capture_mode: str,
    true_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X = prepare_window_frame(raw_df, anomaly_bundle)
    transformed = anomaly_bundle["preprocessor"].transform(X)
    model = anomaly_bundle["model"]
    try:
        scores = -np.asarray(model.decision_function(transformed), dtype=float)
    except Exception:
        scores = -np.asarray(model.decision_function(transformed.toarray()), dtype=float)
    pred = (scores >= threshold).astype(int)
    true_label_name = "Attack" if true_label == 1 else "Normal" if true_label == 0 else None

    docs: list[dict[str, object]] = []
    for record_index, (_, row) in enumerate(raw_df.reset_index(drop=True).iterrows()):
        pred_label = int(pred[record_index])
        docs.append(
            {
                "kind": "live_anomaly_prediction",
                "source": interface,
                "source_file": source_file_tag,
                "window_id": window_id,
                "record_index": record_index,
                "created_at": utc_now(),
                "threshold": threshold,
                "capture_mode": capture_mode,
                "anomaly_score": float(scores[record_index]),
                "anomaly_pred_label": pred_label,
                "anomaly_pred_name": "Anomaly" if pred_label else "Normal",
                "true_label": true_label,
                "true_label_name": true_label_name,
                "correct": int(true_label == pred_label) if true_label is not None else None,
                "raw_csv_path": str(raw_csv),
                **row.to_dict(),
            }
        )

    anomaly_frame = pd.DataFrame(docs)
    return scores, pred, anomaly_frame


def classify_attack_labels(attack_bundle: dict[str, object] | None, raw_df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    if attack_bundle is None or raw_df.empty:
        return None, None
    X = prepare_window_frame(raw_df, attack_bundle)
    transformed = attack_bundle["preprocessor"].transform(X)
    proba_matrix = attack_bundle["model"].predict_proba(transformed)
    pred_idx = np.argmax(proba_matrix, axis=1)
    return proba_matrix.max(axis=1), pred_idx


def attack_class_name(attack_bundle: dict[str, object] | None, index: int | None) -> str | None:
    if index is None:
        return None
    if attack_bundle and "classes_" in attack_bundle:
        classes = list(attack_bundle["classes_"])
        if 0 <= int(index) < len(classes):
            return str(classes[int(index)])
    return str(index)


def assemble_alert_documents(
    *,
    window_id: int,
    interface: str,
    source_file_tag: str,
    capture_mode: str,
    raw_csv: Path,
    raw_df: pd.DataFrame,
    pred: np.ndarray,
    proba: np.ndarray,
    attack_proba: np.ndarray | None,
    attack_pred: np.ndarray | None,
    bundle: dict[str, object],
    alert_df: pd.DataFrame,
    threshold: float,
    true_label: int | None,
) -> list[dict[str, object]]:
    if alert_df.empty:
        return []

    model_version = str(bundle.get("model_version") or bundle.get("active_pointer", {}).get("version") or "binary")
    attack_model_version = str(bundle.get("attack_model_version") or bundle.get("active_pointer", {}).get("version") or model_version)
    rows: list[dict[str, object]] = []
    grouped = alert_df.sort_values(["record_index", "rank"]).groupby("record_index", sort=True)
    for record_index, group in grouped:
        record = raw_df.reset_index(drop=True).iloc[int(record_index)].to_dict() if int(record_index) < len(raw_df) else {}
        pred_label = int(pred[int(record_index)]) if int(record_index) < len(pred) else None
        attack_label_name = None
        attack_label_score = None
        if attack_pred is not None and int(record_index) < len(attack_pred):
            attack_label_name = attack_class_name(bundle.get("attack_bundle"), int(attack_pred[int(record_index)]))
        if attack_proba is not None and int(record_index) < len(attack_proba):
            attack_label_score = float(attack_proba[int(record_index)])
        top_features = [
            {
                "feature": row["feature"],
                "contribution": float(row["contribution"]),
                "abs_contribution": float(row["abs_contribution"]),
                "rank": int(row["rank"]),
            }
            for _, row in group.sort_values("rank").head(10).iterrows()
        ]
        rows.append(
            {
                "kind": "live_alert_explanation",
                "source": interface,
                "source_file": source_file_tag,
                "window_id": window_id,
                "record_index": int(record_index),
                "created_at": utc_now(),
                "prediction_id": f"{window_id}:{int(record_index)}",
                "prediction_kind": "live_prediction",
                "capture_mode": capture_mode,
                "model_version": model_version,
                "attack_model_version": attack_model_version,
                "threshold": float(threshold),
                "raw_csv_path": str(raw_csv),
                "pred_label": pred_label,
                "pred_label_name": "Attack" if pred_label == 1 else "Normal",
                "pred_proba_attack": float(proba[int(record_index)]) if int(record_index) < len(proba) else None,
                "attack_type_name": attack_label_name,
                "attack_type_proba": attack_label_score,
                "true_label": true_label,
                "top_features": top_features,
                "base_value": float(group.iloc[0].get("base_value", 0.0)),
                "expected_value": float(group.iloc[0].get("expected_value", 0.0)),
                "raw_probability": float(proba[int(record_index)]) if int(record_index) < len(proba) else None,
                **record,
            }
        )
    return rows


def append_drift_event(
    *,
    db,
    window_id: int,
    interface: str,
    source_file_tag: str,
    capture_mode: str,
    record_index: int,
    confidence: float,
    adwin_state: dict[str, object],
    trigger_source: str,
) -> dict[str, object]:
    event = {
        "kind": "live_drift_event",
        "source": interface,
        "source_file": source_file_tag,
        "window_id": window_id,
        "record_index": int(record_index),
        "created_at": utc_now(),
        "capture_mode": capture_mode,
        "confidence": float(confidence),
        "trigger_source": trigger_source,
        **adwin_state,
    }
    insert_documents(db[DRIFT_EVENTS_COLLECTION], [event])
    return event


def update_adwin_monitor(
    *,
    adwin: ADWINMonitor,
    scores: np.ndarray,
    db,
    window_id: int,
    interface: str,
    source_file_tag: str,
    capture_mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    drift_events: list[dict[str, object]] = []
    state: dict[str, object] = {}
    for record_index, score in enumerate(scores):
        confidence = float(max(float(score), 1.0 - float(score)))
        state = adwin.update(confidence)
        if state.get("drift_detected"):
            drift_events.append(
                append_drift_event(
                    db=db,
                    window_id=window_id,
                    interface=interface,
                    source_file_tag=source_file_tag,
                    capture_mode=capture_mode,
                    record_index=record_index,
                    confidence=confidence,
                    adwin_state=state,
                    trigger_source="adwin_confidence",
                )
            )
    return state, drift_events


def launch_retrain_job(
    *,
    retrain_script: Path,
    classifier_bundle: Path,
    model_pointer: Path,
    mongo_uri: str,
    db_name: str,
    recent_live_rows: int,
    report_dir: Path,
    edge_csv: Path,
    numeric_threshold: float,
    min_category_count: int,
) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        str(retrain_script),
        "train",
        "--classifier_bundle",
        str(classifier_bundle),
        "--model_out",
        str(model_pointer.with_name("edge_iiot_xgb_model_retrained.joblib")),
        "--report_dir",
        str(report_dir),
        "--edge_csv",
        str(edge_csv),
        "--numeric_threshold",
        str(numeric_threshold),
        "--min_category_count",
        str(min_category_count),
        "--mongo_uri",
        mongo_uri,
        "--db_name",
        db_name,
        "--recent_live_rows",
        str(recent_live_rows),
        "--model_pointer_path",
        str(model_pointer),
    ]
    return subprocess.Popen(command, cwd=str(REPO_ROOT))


def score_live_drift(
    *,
    bundle: dict[str, object],
    reference_raw: pd.DataFrame,
    target_raw: pd.DataFrame,
    interface: str,
    source_file_tag: str,
    window_id: int,
    n_bins: int,
    min_category_count: int,
    numeric_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_scores, summary, _ = compare_batches(
        reference_raw,
        target_raw,
        bundle=bundle,
        mode="live",
        reference_name="dataset_train_split",
        target_name=source_file_tag,
        n_bins=n_bins,
        min_category_count=min_category_count,
        numeric_threshold=numeric_threshold,
    )
    feature_scores = feature_scores.copy()
    feature_scores["importance_type"] = "live_drift"
    feature_scores["source"] = interface
    feature_scores["source_file"] = source_file_tag
    feature_scores["window_id"] = window_id
    feature_scores["capture_mode"] = "live_capture"
    summary = {
        **summary,
        "kind": "live_drift_summary",
        "source": interface,
        "source_file": source_file_tag,
        "window_id": window_id,
        "created_at": utc_now(),
        "capture_mode": "live_capture",
    }
    return feature_scores, summary


def refresh_live_threshold_metrics(
    *,
    db,
    default_threshold: float,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
    min_precision: float = 0.97,
) -> dict[str, object] | None:
    df = collection_to_dataframe(
        db[PREDICTIONS_COLLECTION],
        query={"kind": "live_prediction"},
        projection={"_id": 0},
        sort=[("window_id", 1), ("record_index", 1), ("created_at", 1)],
    )
    if df.empty or "pred_proba_attack" not in df.columns:
        return None

    if "true_label" not in df.columns:
        df["true_label"] = None

    if "source_file" in df.columns:
        inferred_labels = [infer_binary_label_from_name(str(value)) for value in df["source_file"].fillna("")]
    else:
        inferred_labels = [None] * len(df)

    derived_true_label = []
    for existing, inferred in zip(df["true_label"].tolist(), inferred_labels):
        if existing is not None and existing == existing:
            try:
                derived_true_label.append(int(existing))
            except Exception:
                derived_true_label.append(inferred)
        else:
            derived_true_label.append(inferred)

    labeled = df.copy()
    labeled["true_label"] = derived_true_label
    labeled = labeled[labeled["true_label"].notna()].copy()
    if labeled.empty:
        return None

    labeled["true_label"] = pd.to_numeric(labeled["true_label"], errors="coerce")
    labeled["pred_proba_attack"] = pd.to_numeric(labeled["pred_proba_attack"], errors="coerce")
    labeled = labeled.dropna(subset=["true_label", "pred_proba_attack"])
    if labeled.empty:
        return None

    y_true = labeled["true_label"].astype(int).to_numpy()
    y_score = labeled["pred_proba_attack"].astype(float).to_numpy()
    thresholds = build_threshold_grid(threshold_min=threshold_min, threshold_max=threshold_max, threshold_step=threshold_step)
    grid = compute_threshold_metrics(y_true, y_score, thresholds)
    if grid.empty:
        return None

    latest_window_id = int(labeled["window_id"].max()) if "window_id" in labeled.columns and labeled["window_id"].notna().any() else None
    grid = grid.copy()
    grid["kind"] = "threshold_grid"
    grid["source"] = "live"
    grid["source_name"] = "live"
    grid["created_at"] = utc_now()
    grid["updated_at"] = utc_now()
    grid["window_id"] = latest_window_id
    grid["sample_count"] = int(len(labeled))
    grid["label_source"] = "inferred_from_source_file"
    upsert_documents(db[THRESHOLD_METRICS_COLLECTION], grid.to_dict(orient="records"), ["kind", "source", "threshold"])

    recommendations = select_recommendations(grid, min_precision=min_precision, default_threshold=default_threshold)
    summary_text = "\n".join(
        [
            "# Threshold Calibration Summary",
            "",
            "Live threshold calibration is computed from labeled live rows currently available in MongoDB.",
            f"- Samples: {len(labeled)}",
            f"- Latest window_id: {latest_window_id if latest_window_id is not None else 'n/a'}",
            f"- Threshold grid: {threshold_min:.2f} to {threshold_max:.2f} step {threshold_step:.2f}",
            f"- Default threshold: {default_threshold:.2f}",
            f"- Max F1 threshold: {recommendations['recommended']['max_f1']['threshold']:.2f}",
            f"- Max F2 threshold: {recommendations['recommended']['max_f2']['threshold']:.2f}",
            f"- Best recall under precision constraint: {recommendations['recommended']['highest_recall_under_min_precision']['threshold']:.2f}",
            f"- Lowest FNR under precision constraint: {recommendations['recommended']['lowest_fnr_under_min_precision']['threshold']:.2f}",
        ]
    )
    summary_doc = {
        "kind": "threshold_calibration_summary",
        "source": "live",
        "artifact_path": "mongo://live_threshold_metrics",
        "created_at": utc_now(),
        "payload": {
            "text": summary_text,
            "source_name": "live",
            "sample_count": int(len(labeled)),
            "latest_window_id": latest_window_id,
            "recommendations": recommendations,
            "threshold_grid": {
                "min": float(threshold_min),
                "max": float(threshold_max),
                "step": float(threshold_step),
            },
        },
    }
    upsert_documents(db[ANALYSIS_SUMMARIES_COLLECTION], [summary_doc], ["kind", "artifact_path"])
    return summary_doc


def score_window(
    *,
    window_id: int,
    raw_csv: Path,
    bundle: dict[str, object],
    threshold: float,
    shap_sample_rows: int,
    shap_top_n: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, pd.DataFrame, pd.DataFrame]:
    if not raw_csv.exists() or raw_csv.stat().st_size == 0:
        empty = pd.DataFrame()
        return empty, np.array([]), np.array([], dtype=int), None, None, empty, empty

    df = pd.read_csv(raw_csv, low_memory=False)
    df = normalize_columns(df)
    if df.empty:
        empty = pd.DataFrame()
        return empty, np.array([]), np.array([], dtype=int), None, None, empty, empty

    X = prepare_window_frame(df, bundle)
    transformed = bundle["preprocessor"].transform(X)
    proba = bundle["model"].predict_proba(transformed)[:, 1]
    pred = (proba >= threshold).astype(int)
    attack_proba, attack_pred = classify_attack_labels(bundle.get("attack_bundle"), df)
    feature_names = get_transformed_feature_names(bundle["preprocessor"])
    shap_df = safe_mean_absolute_contrib(
        transformed,
        bundle["model"],
        feature_names,
        sample_rows=shap_sample_rows,
    )
    shap_df = shap_df.head(15).copy()
    shap_df["window_id"] = window_id
    alert_df = safe_alert_contribs(
        transformed,
        bundle["model"],
        feature_names,
        sample_rows=shap_sample_rows,
        top_n=shap_top_n,
    )
    alert_df["window_id"] = window_id
    return df, proba, pred, attack_proba, attack_pred, shap_df, alert_df


def infer_window_true_label(*, source_file_tag: str, injected_pcap_path: str | None = None) -> int | None:
    candidates: list[str] = []
    if injected_pcap_path:
        candidates.append(Path(injected_pcap_path).name)
    if source_file_tag:
        candidates.append(source_file_tag)
    for candidate in candidates:
        label = infer_binary_label_from_name(candidate)
        if label is not None:
            return label
    return None


def prediction_documents(
    *,
    window_id: int,
    interface: str,
    raw_df: pd.DataFrame,
    proba: np.ndarray,
    pred: np.ndarray,
    attack_proba: np.ndarray | None,
    attack_pred: np.ndarray | None,
    threshold: float,
    raw_csv: Path,
    source_file_tag: str,
    capture_mode: str,
    model_version: str | None = None,
    attack_model_version: str | None = None,
    attack_bundle: dict[str, object] | None = None,
    true_label: int | None = None,
) -> list[dict[str, object]]:
    true_label_name = "Attack" if true_label == 1 else "Normal" if true_label == 0 else None
    docs: list[dict[str, object]] = []
    for record_index, (_, row) in enumerate(raw_df.reset_index(drop=True).iterrows()):
        pred_label = int(pred[record_index])
        attack_type_name = None
        attack_type_proba = None
        if attack_pred is not None and record_index < len(attack_pred):
            attack_type_name = attack_class_name(attack_bundle, int(attack_pred[record_index]))
        if attack_proba is not None and record_index < len(attack_proba):
            attack_type_proba = float(attack_proba[record_index])
        payload = {
            "kind": "live_prediction",
            "source": interface,
            "source_file": source_file_tag,
            "window_id": window_id,
            "record_index": record_index,
            "created_at": utc_now(),
            "threshold": threshold,
            "capture_mode": capture_mode,
            "model_version": model_version,
            "attack_model_version": attack_model_version,
            "pred_proba_attack": float(proba[record_index]),
            "pred_label": pred_label,
            "pred_label_name": "Attack" if pred_label else "Normal",
            "attack_type_name": attack_type_name,
            "attack_type_proba": attack_type_proba,
            "true_label": true_label,
            "true_label_name": true_label_name,
            "correct": int(true_label == pred_label) if true_label is not None else None,
            "raw_csv_path": str(raw_csv),
            **row.to_dict(),
        }
        docs.append(payload)
    return docs


def packet_documents(
    *,
    window_id: int,
    interface: str,
    kind: str,
    raw_df: pd.DataFrame,
    raw_csv: Path,
    source_file_tag: str,
    true_label: int | None = None,
) -> list[dict[str, object]]:
    true_label_name = "Attack" if true_label == 1 else "Normal" if true_label == 0 else None
    return dataframe_to_documents(
        raw_df,
        kind=kind,
        source=interface,
        source_file=source_file_tag,
        extra_fields={
            "window_id": window_id,
            "raw_csv_path": str(raw_csv),
            "true_label": true_label,
            "true_label_name": true_label_name,
        },
    )


def live_window_summary(
    *,
    window_id: int,
    interface: str,
    source_file_tag: str,
    capture_mode: str,
    raw_csv: Path,
    window_start: datetime,
    window_end: datetime,
    df: pd.DataFrame,
    proba: np.ndarray,
    pred: np.ndarray,
    attack_proba: np.ndarray | None,
    threshold: float,
    file_max_threshold: float,
    file_ratio_threshold: float,
    min_records: int,
    unsupported_fields: list[str],
    capture_filter: str | None,
    display_filter: str | None,
    packet_count: int | None,
    feature_importance_rows: pd.DataFrame,
    model_version: str | None = None,
    attack_model_version: str | None = None,
    adwin_state: dict[str, object] | None = None,
    injected_pcap_path: str | None = None,
) -> dict[str, object]:
    records = int(len(df))
    attack_records = int(pred.sum()) if len(pred) else 0
    attack_ratio = float(attack_records / records) if records else 0.0
    summary = {
        "kind": "live_window_summary",
        "source": interface,
        "source_file": source_file_tag,
        "window_id": window_id,
        "created_at": window_end,
        "window_start": window_start,
        "window_end": window_end,
        "interface": interface,
        "capture_mode": capture_mode,
        "records": records,
        "attack_records": attack_records,
        "attack_record_ratio": attack_ratio,
        "mean_attack_probability": float(np.mean(proba)) if len(proba) else 0.0,
        "median_attack_probability": float(np.median(proba)) if len(proba) else 0.0,
        "p95_attack_probability": float(np.quantile(proba, 0.95)) if len(proba) else 0.0,
        "max_attack_probability": float(np.max(proba)) if len(proba) else 0.0,
        "mean_attack_type_probability": float(np.mean(attack_proba)) if attack_proba is not None and len(attack_proba) else None,
        "threshold": threshold,
        "model_version": model_version,
        "attack_model_version": attack_model_version,
        "adwin_state": adwin_state or {},
        "file_max_threshold": file_max_threshold,
        "file_ratio_threshold": file_ratio_threshold,
        "window_pred_label": int(
            records >= min_records
            and float(np.max(proba)) >= file_max_threshold
            and attack_ratio >= file_ratio_threshold
        ),
        "unsupported_tshark_fields": "|".join(unsupported_fields),
        "capture_filter": capture_filter,
        "display_filter": display_filter,
        "packet_count": packet_count,
        "raw_csv_path": str(raw_csv),
        "injected_pcap_path": injected_pcap_path,
        "feature_count": int(df.shape[1]),
        "shap_feature_count": int(len(feature_importance_rows)),
        "shap_available": bool(len(feature_importance_rows)),
    }
    return summary


def score_and_store_window(
    *,
    db,
    window_id: int,
    interface: str,
    bundle: dict[str, object],
    raw_csv: Path,
    df: pd.DataFrame,
    proba: np.ndarray,
    pred: np.ndarray,
    attack_proba: np.ndarray | None,
    attack_pred: np.ndarray | None,
    threshold: float,
    fields: list[str],
    unsupported_fields: list[str],
    raw_docs_kind: str,
    feature_docs_kind: str,
    source_file_tag: str,
    capture_mode: str,
    window_start: datetime,
    window_end: datetime,
    file_max_threshold: float,
    file_ratio_threshold: float,
    min_records: int,
    capture_filter: str | None,
    display_filter: str | None,
    packet_count: int | None,
    shap_df: pd.DataFrame,
    alert_df: pd.DataFrame,
    anomaly_bundle: dict[str, object] | None,
    drift_reference_raw: pd.DataFrame | None,
    drift_n_bins: int,
    drift_min_category_count: int,
    numeric_threshold: float,
    tshark_timeout_seconds: int | None = None,
    injected_pcap_path: str | None = None,
    model_version: str | None = None,
    attack_model_version: str | None = None,
    adwin_state: dict[str, object] | None = None,
) -> dict[str, object]:
    window_true_label = infer_window_true_label(source_file_tag=source_file_tag, injected_pcap_path=injected_pcap_path)
    raw_docs = packet_documents(
        window_id=window_id,
        interface=interface,
        kind=raw_docs_kind,
        raw_df=df,
        raw_csv=raw_csv,
        source_file_tag=source_file_tag,
        true_label=window_true_label,
    )
    X = prepare_window_frame(df, bundle)
    feature_docs = packet_documents(
        window_id=window_id,
        interface=interface,
        kind=feature_docs_kind,
        raw_df=X,
        raw_csv=raw_csv,
        source_file_tag=source_file_tag,
        true_label=window_true_label,
    )
    prediction_docs = prediction_documents(
        window_id=window_id,
        interface=interface,
        raw_df=df,
        proba=proba,
        pred=pred,
        attack_proba=attack_proba,
        attack_pred=attack_pred,
        threshold=threshold,
        raw_csv=raw_csv,
        source_file_tag=source_file_tag,
        capture_mode=capture_mode,
        model_version=model_version,
        attack_model_version=attack_model_version,
        attack_bundle=bundle.get("attack_bundle"),
        true_label=window_true_label,
    )
    for doc in prediction_docs:
        doc["raw_csv_path"] = str(raw_csv)
    insert_documents(db[RAW_PACKETS_COLLECTION], raw_docs)
    insert_documents(db[FEATURE_VECTORS_COLLECTION], feature_docs)
    insert_documents(db[PREDICTIONS_COLLECTION], prediction_docs)
    if not shap_df.empty:
        alert_docs = assemble_alert_documents(
            window_id=window_id,
            interface=interface,
            source_file_tag=source_file_tag,
            capture_mode=capture_mode,
            raw_csv=raw_csv,
            raw_df=df,
            pred=pred,
            proba=proba,
            attack_proba=attack_proba,
            attack_pred=attack_pred,
            bundle=bundle,
            alert_df=alert_df,
            threshold=threshold,
            true_label=window_true_label,
        )
        if alert_docs:
            insert_documents(db[ALERT_EXPLANATIONS_COLLECTION], alert_docs)
    if not shap_df.empty:
        live_shap_docs = shap_df.copy()
        live_shap_docs["importance_type"] = "live_shap"
        live_shap_docs["source"] = interface
        live_shap_docs["source_file"] = source_file_tag
        live_shap_docs["window_id"] = window_id
        live_shap_docs["capture_mode"] = capture_mode
        insert_documents(db[FEATURE_IMPORTANCE_COLLECTION], live_shap_docs.to_dict(orient="records"))

    summary = live_window_summary(
        window_id=window_id,
        interface=interface,
        source_file_tag=source_file_tag,
        capture_mode=capture_mode,
        raw_csv=raw_csv,
        window_start=window_start,
        window_end=window_end,
        df=df,
        proba=proba,
        pred=pred,
        attack_proba=attack_proba,
        threshold=threshold,
        file_max_threshold=file_max_threshold,
        file_ratio_threshold=file_ratio_threshold,
        min_records=min_records,
        unsupported_fields=unsupported_fields,
        capture_filter=capture_filter,
        display_filter=display_filter,
        packet_count=packet_count,
        feature_importance_rows=shap_df,
        model_version=model_version,
        attack_model_version=attack_model_version,
        adwin_state=adwin_state,
        injected_pcap_path=injected_pcap_path,
    )
    upsert_documents(db[LIVE_WINDOWS_COLLECTION], [summary], ["kind", "source_file", "window_id"])

    anomaly_records = 0
    anomaly_ratio = 0.0
    anomaly_mean = 0.0
    anomaly_median = 0.0
    anomaly_p95 = 0.0
    anomaly_max = 0.0
    if anomaly_bundle is not None:
        anomaly_threshold = float(anomaly_bundle.get("threshold", 0.0))
        anomaly_scores, anomaly_pred, anomaly_df = score_anomaly_window(
            raw_df=df,
            raw_csv=raw_csv,
            anomaly_bundle=anomaly_bundle,
            threshold=anomaly_threshold,
            interface=interface,
            source_file_tag=source_file_tag,
            window_id=window_id,
            capture_mode=capture_mode,
            true_label=window_true_label,
        )
        insert_documents(db[PREDICTIONS_COLLECTION], anomaly_df.to_dict(orient="records"))
        anomaly_records = int(anomaly_pred.sum()) if len(anomaly_pred) else 0
        anomaly_ratio = float(anomaly_records / len(anomaly_pred)) if len(anomaly_pred) else 0.0
        anomaly_mean = float(np.mean(anomaly_scores)) if len(anomaly_scores) else 0.0
        anomaly_median = float(np.median(anomaly_scores)) if len(anomaly_scores) else 0.0
        anomaly_p95 = float(np.quantile(anomaly_scores, 0.95)) if len(anomaly_scores) else 0.0
        anomaly_max = float(np.max(anomaly_scores)) if len(anomaly_scores) else 0.0

    if drift_reference_raw is not None and not df.empty:
        live_drift_scores, live_drift_summary = score_live_drift(
            bundle=bundle,
            reference_raw=drift_reference_raw,
            target_raw=df,
            interface=interface,
            source_file_tag=source_file_tag,
            window_id=window_id,
            n_bins=drift_n_bins,
            min_category_count=drift_min_category_count,
            numeric_threshold=numeric_threshold,
        )
        insert_documents(db[FEATURE_IMPORTANCE_COLLECTION], live_drift_scores.to_dict(orient="records"))
        insert_documents(db[ANALYSIS_SUMMARIES_COLLECTION], [live_drift_summary])
    refresh_live_threshold_metrics(db=db, default_threshold=threshold)
    summary.update(
        {
            "anomaly_records": anomaly_records,
            "anomaly_record_ratio": anomaly_ratio,
            "mean_anomaly_score": anomaly_mean,
            "median_anomaly_score": anomaly_median,
            "p95_anomaly_score": anomaly_p95,
            "max_anomaly_score": anomaly_max,
        }
    )
    upsert_documents(db[LIVE_WINDOWS_COLLECTION], [summary], ["kind", "source_file", "window_id"])
    return summary


def process_injection_queue(
    *,
    db,
    status_path: Path,
    base_status: dict[str, object],
    bundle_path: Path,
    anomaly_bundle: dict[str, object] | None,
    drift_reference_raw: pd.DataFrame | None,
    adwin: ADWINMonitor,
    trigger_retrain,
    tshark: str,
    injection_dir: Path,
    processed_dir: Path,
    capture_dir: Path,
    interface: str,
    fields: list[str],
    valid_fields: set[str] | None,
    threshold: float,
    file_max_threshold: float,
    file_ratio_threshold: float,
    min_records: int,
    display_filter: str | None,
    packet_count: int | None,
    shap_sample_rows: int,
    shap_top_n: int,
    drift_n_bins: int,
    drift_min_category_count: int,
    numeric_threshold: float,
    tshark_timeout_seconds: int | None,
    window_id: int,
    model_version: str | None,
    attack_model_version: str | None,
) -> int:
    processed = 0
    for pcap_path in injection_pcap_files(injection_dir):
        window_id += 1
        processed += 1
        source_file_tag = f"{live_window_tag(window_id)}_{pcap_path.stem}"
        raw_csv = capture_dir / f"{source_file_tag}.csv"
        start_time = utc_now()
        bundle = load_bundle(bundle_path)
        model_version = str(bundle.get("model_version") or bundle.get("active_pointer", {}).get("version") or model_version or "binary")
        attack_model_version = str(bundle.get("attack_model_version") or bundle.get("active_pointer", {}).get("version") or attack_model_version or "binary")
        _, unsupported_fields = run_pcap_to_csv(
            tshark=tshark,
            input_pcap=pcap_path,
            output_csv=raw_csv,
            fields=fields,
            valid_fields=valid_fields,
            display_filter=display_filter,
            packet_count=packet_count,
            timeout_seconds=tshark_timeout_seconds,
        )
        df, proba, pred, attack_proba, attack_pred, shap_df, alert_df = score_window(
            window_id=window_id,
            raw_csv=raw_csv,
            bundle=bundle,
            threshold=threshold,
            shap_sample_rows=shap_sample_rows,
            shap_top_n=shap_top_n,
        )
        adwin_state, drift_events = update_adwin_monitor(
            adwin=adwin,
            scores=proba,
            db=db,
            window_id=window_id,
            interface=interface,
            source_file_tag=source_file_tag,
            capture_mode="pcap_injection",
        )
        if drift_events and trigger_retrain is not None:
            trigger_retrain(
                f"adwin_drift_{len(drift_events)}",
                source_file_tag=source_file_tag,
                window_id=window_id,
            )
        archived_pcap = archive_injected_pcap(pcap_path, processed_dir, window_id=window_id)
        summary = score_and_store_window(
            db=db,
            window_id=window_id,
            interface=interface,
            bundle=bundle,
            raw_csv=raw_csv,
            df=df,
            proba=proba,
            pred=pred,
            attack_proba=attack_proba,
            attack_pred=attack_pred,
            threshold=threshold,
            fields=fields,
            unsupported_fields=unsupported_fields,
            raw_docs_kind="live_raw_packet",
            feature_docs_kind="live_feature_vector",
            source_file_tag=source_file_tag,
            capture_mode="pcap_injection",
            window_start=start_time,
            window_end=utc_now(),
            file_max_threshold=file_max_threshold,
            file_ratio_threshold=file_ratio_threshold,
            min_records=min_records,
            capture_filter=None,
            display_filter=display_filter,
            packet_count=packet_count,
            shap_df=shap_df,
            alert_df=alert_df,
            anomaly_bundle=anomaly_bundle,
            drift_reference_raw=drift_reference_raw,
            drift_n_bins=drift_n_bins,
            drift_min_category_count=drift_min_category_count,
            numeric_threshold=numeric_threshold,
            injected_pcap_path=str(archived_pcap),
            model_version=model_version,
            attack_model_version=attack_model_version,
            adwin_state=adwin_state,
        )
        print(
            f"[inject {window_id}] pcap={pcap_path.name} records={summary['records']} "
            f"attack_ratio={summary['attack_record_ratio']:.4f} max_proba={summary['max_attack_probability']:.4f} "
            f"pred_label={summary['window_pred_label']}"
        )
        write_live_status_success(
            status_path=status_path,
            base_status=base_status,
            last_window_id=window_id,
            last_summary=summary,
            extra={"last_injected_pcap": str(archived_pcap)},
        )
    return window_id


def record_loop(args: argparse.Namespace) -> None:
    bundle_path = Path(args.model_path)
    bundle = load_bundle(bundle_path)
    anomaly_bundle = load_anomaly_bundle(Path(args.anomaly_model_path))
    threshold = float(args.threshold if args.threshold is not None else bundle["threshold"])
    tshark = find_tshark(args.tshark)
    interface_ok, interface_error = validate_capture_interface(tshark, args.interface)
    if not interface_ok:
        status = {
            "state": "failed",
            "started_at": utc_now().isoformat(),
            "interface": args.interface,
            "tshark": tshark,
            "pid": os.getpid(),
            "last_error": interface_error,
            "model_path": str(bundle_path),
        }
        write_live_status(status_path=Path(args.status_path), base_status=status, last_window_id=0, state="failed")
        print(f"ERROR: {interface_error}", file=sys.stderr)
        raise SystemExit(1)
    valid_fields = available_tshark_fields(tshark) if args.validate_tshark_fields else None
    fields = bundle_field_contract(bundle, include_metadata=not args.no_metadata)
    drift_reference_box: dict[str, pd.DataFrame | None] = {"value": None}
    adwin = ADWINMonitor(delta=args.adwin_delta, min_window=args.adwin_min_window, max_window=args.adwin_max_window)
    retrain_lock = threading.Lock()
    retrain_state: dict[str, object] = {
        "running": False,
        "last_started_at": None,
        "last_finished_at": None,
        "last_exit_code": None,
        "last_result": None,
        "last_process_pid": None,
    }

    mongo_uri = args.mongo_uri
    db_name = args.db_name
    db = connect_database(mongo_uri, db_name)
    ensure_indexes(db)

    output_dir = Path(args.output_dir)
    capture_dir = Path(args.capture_dir)
    injection_dir = Path(args.injection_dir)
    processed_dir = Path(args.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    injection_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "state": "starting",
        "started_at": utc_now().isoformat(),
        "interface": args.interface,
        "tshark": tshark,
        "window_seconds": args.window_seconds,
        "packet_count": args.packet_count,
        "mongo_uri": mongo_uri,
        "db_name": db_name,
        "pid": os.getpid(),
        "injection_dir": str(injection_dir),
        "live_anomaly_enabled": bool(anomaly_bundle is not None),
        "live_drift_enabled": False,
        "model_path": str(bundle_path),
        "model_version": str(bundle.get("model_version") or bundle.get("active_pointer", {}).get("version") or "binary"),
        "attack_model_version": str(bundle.get("attack_model_version") or bundle.get("active_pointer", {}).get("version") or "binary"),
        "adwin_delta": float(args.adwin_delta),
        "auto_retrain": bool(args.auto_retrain),
    }
    previous_status = read_status(Path(args.status_path))
    previous_window_id = int(previous_status.get("last_window_id", 0) or 0)
    previous_live_summary = db[LIVE_WINDOWS_COLLECTION].find_one(
        {"kind": "live_window_summary"},
        sort=[("window_id", -1), ("created_at", -1)],
        projection={"_id": 0, "window_id": 1},
    )
    previous_live_window_id = int(previous_live_summary.get("window_id", 0) or 0) if previous_live_summary else 0
    last_window_id = max(previous_window_id, previous_live_window_id)
    status["last_window_id"] = last_window_id
    write_live_status(status_path=Path(args.status_path), base_status=status, last_window_id=last_window_id)

    def trigger_retrain(reason: str, *, source_file_tag: str, window_id: int) -> None:
        if not args.auto_retrain:
            return
        with retrain_lock:
            if retrain_state["running"]:
                return

            retrain_state["running"] = True
            retrain_state["last_started_at"] = utc_now().isoformat()
            request_doc = {
                "kind": "live_retrain_request",
                "source": args.interface,
                "source_file": source_file_tag,
                "window_id": window_id,
                "created_at": utc_now(),
                "reason": reason,
                "model_path": str(bundle_path),
                "model_version": status.get("model_version"),
                "attack_model_version": status.get("attack_model_version"),
                "adwin_state": json_safe({**adwin.state.__dict__, "window": list(adwin.state.window)}),
                "auto_retrain": True,
            }
            insert_documents(db[RETRAIN_EVENTS_COLLECTION], [request_doc])

        def _run_retrain() -> None:
            try:
                proc = launch_retrain_job(
                    retrain_script=REPO_ROOT / "src" / "edge_iiot_retrain.py",
                    classifier_bundle=bundle_path,
                    model_pointer=Path(args.model_pointer_path),
                    mongo_uri=mongo_uri,
                    db_name=db_name,
                    recent_live_rows=args.retrain_recent_rows,
                    report_dir=Path(args.output_dir).parent / "reports",
                    edge_csv=Path(DEFAULT_EDGE_CSV),
                    numeric_threshold=args.numeric_threshold,
                    min_category_count=args.retrain_min_category_count,
                )
                retrain_state["last_process_pid"] = int(proc.pid)
                exit_code = proc.wait()
                retrain_state["last_exit_code"] = int(exit_code)
                retrain_state["last_finished_at"] = utc_now().isoformat()
                result_doc = {
                    "kind": "live_retrain_result",
                    "source": args.interface,
                    "source_file": source_file_tag,
                    "window_id": window_id,
                    "created_at": utc_now(),
                    "reason": reason,
                    "exit_code": int(exit_code),
                    "model_pointer_path": str(args.model_pointer_path),
                    "model_path": str(bundle_path),
                    "accepted": bool(exit_code == 0),
                }
                insert_documents(db[RETRAIN_EVENTS_COLLECTION], [result_doc])
                status_update = {
                    "last_retrain_reason": reason,
                    "last_retrain_exit_code": int(exit_code),
                    "last_retrain_finished_at": retrain_state["last_finished_at"],
                }
                write_live_status(
                    status_path=Path(args.status_path),
                    base_status={**status, **status_update},
                    last_window_id=window_id,
                    extra={"retrain_state": json_safe(retrain_state)},
                )
            except Exception as exc:
                retrain_state["last_exit_code"] = 1
                retrain_state["last_finished_at"] = utc_now().isoformat()
                insert_documents(
                    db[RETRAIN_EVENTS_COLLECTION],
                    [
                        {
                            "kind": "live_retrain_result",
                            "source": args.interface,
                            "source_file": source_file_tag,
                            "window_id": window_id,
                            "created_at": utc_now(),
                            "reason": reason,
                            "exit_code": 1,
                            "error": str(exc),
                            "model_pointer_path": str(args.model_pointer_path),
                            "model_path": str(bundle_path),
                            "accepted": False,
                        }
                    ],
                )
            finally:
                with retrain_lock:
                    retrain_state["running"] = False

        threading.Thread(target=_run_retrain, daemon=True).start()

    if not args.disable_live_drift:
        def _load_drift_reference() -> None:
            try:
                drift_reference_raw, _, _ = load_dataset_batches(DEFAULT_EDGE_CSV)
                drift_reference_box["value"] = drift_reference_raw
                status["live_drift_enabled"] = True
                status["last_error"] = None
                write_live_status(
                    status_path=Path(args.status_path),
                    base_status=status,
                    last_window_id=last_window_id,
                    extra={"live_drift_enabled": True, "last_error": None},
                )
            except Exception as exc:
                status["live_drift_enabled"] = False
                status["last_error"] = f"drift load failed: {exc}"
                write_live_status(
                    status_path=Path(args.status_path),
                    base_status=status,
                    last_window_id=last_window_id,
                    extra={"live_drift_enabled": False, "last_error": f"drift load failed: {exc}"},
                )

        threading.Thread(target=_load_drift_reference, daemon=True).start()

    refresh_live_threshold_metrics(db=db, default_threshold=threshold)

    print(f"tshark      : {tshark}")
    print(f"interface   : {args.interface}")
    print(f"window_sec  : {args.window_seconds}")
    print(f"packet_count: {args.packet_count}")
    print(f"threshold   : {threshold}")
    print(f"mongo db    : {db_name}")
    print(f"injection   : {injection_dir}")
    print(f"anomaly     : {str(Path(args.anomaly_model_path)) if anomaly_bundle is not None else 'disabled'}")
    print(f"live drift  : {'background load enabled' if not args.disable_live_drift else 'disabled'}")

    try:
        current_bundle = lambda: load_bundle(bundle_path)
        injection_common_kwargs = {
            "anomaly_bundle": anomaly_bundle,
            "tshark": tshark,
            "injection_dir": injection_dir,
            "processed_dir": processed_dir,
            "capture_dir": capture_dir,
            "interface": args.interface,
            "fields": fields,
            "valid_fields": valid_fields,
            "threshold": threshold,
            "file_max_threshold": args.file_max_threshold,
            "file_ratio_threshold": args.file_ratio_threshold,
            "min_records": args.min_records,
            "display_filter": args.display_filter,
            "packet_count": args.packet_count,
            "shap_sample_rows": args.shap_sample_rows,
            "shap_top_n": args.shap_top_n,
            "drift_n_bins": args.drift_n_bins,
            "drift_min_category_count": args.drift_min_category_count,
            "numeric_threshold": args.numeric_threshold,
            "tshark_timeout_seconds": args.tshark_timeout_seconds,
            "adwin": adwin,
            "trigger_retrain": trigger_retrain,
            "model_version": status["model_version"],
            "attack_model_version": status["attack_model_version"],
        }

        def build_injection_kwargs() -> dict[str, object]:
            return {
                **injection_common_kwargs,
                "drift_reference_raw": drift_reference_box["value"],
            }

        while True:
            try:
                last_window_id = process_injection_queue(
                    db=db,
                    status_path=Path(args.status_path),
                    base_status=status,
                    bundle_path=bundle_path,
                    **build_injection_kwargs(),
                    window_id=last_window_id,
                )
                current_window_id = last_window_id + 1
                if args.max_windows and args.max_windows > 0 and current_window_id > args.max_windows:
                    break

                bundle = current_bundle()
                status["model_version"] = str(bundle.get("model_version") or bundle.get("active_pointer", {}).get("version") or "binary")
                status["attack_model_version"] = str(bundle.get("attack_model_version") or bundle.get("active_pointer", {}).get("version") or "binary")
                source_file_tag = live_window_tag(current_window_id)
                raw_csv = capture_dir / f"{source_file_tag}.csv"
                start_time = utc_now()
                try:
                    supported_fields, unsupported_fields = run_capture_window_to_csv(
                        tshark=tshark,
                        interface=args.interface,
                        duration_seconds=args.window_seconds,
                        output_csv=raw_csv,
                        fields=fields,
                        valid_fields=valid_fields,
                        capture_filter=args.capture_filter,
                        display_filter=args.display_filter,
                        packet_count=args.packet_count,
                        timeout_seconds=args.tshark_timeout_seconds,
                    )
                except Exception as exc:
                    print(f"[{current_window_id}] capture error: {exc}")
                    write_live_status(
                        status_path=Path(args.status_path),
                        base_status=status,
                        last_window_id=last_window_id,
                        extra={"last_error": str(exc)},
                    )
                    if args.pause_seconds > 0:
                        time.sleep(args.pause_seconds)
                    continue

                df, proba, pred, attack_proba, attack_pred, shap_df, alert_df = score_window(
                    window_id=current_window_id,
                    raw_csv=raw_csv,
                    bundle=bundle,
                    threshold=threshold,
                    shap_sample_rows=args.shap_sample_rows,
                    shap_top_n=args.shap_top_n,
                )
                if df.empty:
                    print(f"[{current_window_id}] empty capture window")
                    summary = {
                        "kind": "live_window_summary",
                        "source": args.interface,
                        "source_file": source_file_tag,
                        "window_id": current_window_id,
                        "created_at": utc_now(),
                        "window_start": start_time,
                        "window_end": utc_now(),
                        "interface": args.interface,
                        "capture_mode": "live_capture",
                        "records": 0,
                        "attack_records": 0,
                        "attack_record_ratio": 0.0,
                        "mean_attack_probability": 0.0,
                        "median_attack_probability": 0.0,
                        "p95_attack_probability": 0.0,
                        "max_attack_probability": 0.0,
                        "threshold": threshold,
                        "file_max_threshold": args.file_max_threshold,
                        "file_ratio_threshold": args.file_ratio_threshold,
                        "window_pred_label": 0,
                        "unsupported_tshark_fields": "|".join(unsupported_fields),
                        "capture_filter": args.capture_filter,
                        "display_filter": args.display_filter,
                        "packet_count": args.packet_count,
                        "raw_csv_path": str(raw_csv),
                        "feature_count": int(len(fields)),
                        "shap_feature_count": 0,
                        "shap_available": False,
                        "supported_tshark_fields": "|".join(supported_fields),
                        "injected_pcap_path": None,
                        "model_version": status["model_version"],
                        "attack_model_version": status["attack_model_version"],
                        "adwin_state": json_safe(adwin.state.__dict__),
                    }
                    insert_documents(db[LIVE_WINDOWS_COLLECTION], [summary])
                    write_live_status_success(
                        status_path=Path(args.status_path),
                        base_status=status,
                        last_window_id=current_window_id,
                        last_summary=summary,
                    )
                    last_window_id = current_window_id
                    continue

                X = prepare_window_frame(df, bundle)
                adwin_state, drift_events = update_adwin_monitor(
                    adwin=adwin,
                    scores=proba,
                    db=db,
                    window_id=current_window_id,
                    interface=args.interface,
                    source_file_tag=source_file_tag,
                    capture_mode="live_capture",
                )
                if drift_events and args.auto_retrain:
                    trigger_retrain(
                        f"adwin_drift_{len(drift_events)}",
                        source_file_tag=source_file_tag,
                        window_id=current_window_id,
                    )
                summary = score_and_store_window(
                    db=db,
                    window_id=current_window_id,
                    interface=args.interface,
                    bundle=bundle,
                    raw_csv=raw_csv,
                    df=df,
                    proba=proba,
                    pred=pred,
                    attack_proba=attack_proba,
                    attack_pred=attack_pred,
                    threshold=threshold,
                    fields=fields,
                    unsupported_fields=unsupported_fields,
                    raw_docs_kind="live_raw_packet",
                    feature_docs_kind="live_feature_vector",
                    source_file_tag=source_file_tag,
                    capture_mode="live_capture",
                    window_start=start_time,
                    window_end=utc_now(),
                    file_max_threshold=args.file_max_threshold,
                    file_ratio_threshold=args.file_ratio_threshold,
                    min_records=args.min_records,
                    capture_filter=args.capture_filter,
                    display_filter=args.display_filter,
                    packet_count=args.packet_count,
                    shap_df=shap_df,
                    alert_df=alert_df,
                    anomaly_bundle=anomaly_bundle,
                    drift_reference_raw=drift_reference_box["value"],
                    drift_n_bins=args.drift_n_bins,
                    drift_min_category_count=args.drift_min_category_count,
                    numeric_threshold=args.numeric_threshold,
                    tshark_timeout_seconds=args.tshark_timeout_seconds,
                    injected_pcap_path=None,
                    model_version=status["model_version"],
                    attack_model_version=status["attack_model_version"],
                    adwin_state=adwin_state,
                )

                print(
                    f"[{current_window_id}] records={summary['records']} attack_ratio={summary['attack_record_ratio']:.4f} "
                    f"max_proba={summary['max_attack_probability']:.4f} pred_label={summary['window_pred_label']}"
                )
                write_live_status_success(
                    status_path=Path(args.status_path),
                    base_status=status,
                    last_window_id=current_window_id,
                    last_summary=summary,
                )

                last_window_id = current_window_id
                if args.pause_seconds > 0:
                    time.sleep(args.pause_seconds)
                last_window_id = process_injection_queue(
                    db=db,
                    status_path=Path(args.status_path),
                    base_status=status,
                    bundle_path=bundle_path,
                    **build_injection_kwargs(),
                    window_id=last_window_id,
                )
            except Exception as exc:
                print(f"Live loop error: {exc}")
                write_live_status(
                    status_path=Path(args.status_path),
                    base_status=status,
                    last_window_id=last_window_id,
                    extra={"last_error": str(exc)},
                )
                if args.pause_seconds > 0:
                    time.sleep(args.pause_seconds)
                continue
    except KeyboardInterrupt:
        print("Live capture stopped by user.")
    finally:
        write_status(
            Path(args.status_path),
            {
                **status,
                "state": "stopped",
                "stopped_at": utc_now().isoformat(),
                "last_window_id": last_window_id,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Edge-IIoT capture and scoring worker.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live_parser = subparsers.add_parser("live", help="Capture live windows, score them, and write to MongoDB.")
    live_parser.add_argument("--model_path", default=str(DEFAULT_MODEL_POINTER_PATH))
    live_parser.add_argument("--anomaly_model_path", default=str(DEFAULT_ANOMALY_MODEL_PATH))
    live_parser.add_argument("--interface", required=True)
    live_parser.add_argument("--tshark", default=None)
    live_parser.add_argument("--window_seconds", type=int, default=30)
    live_parser.add_argument("--packet_count", type=int, default=None)
    live_parser.add_argument("--capture_filter", default=None)
    live_parser.add_argument("--display_filter", default=None)
    live_parser.add_argument("--max_windows", type=int, default=0)
    live_parser.add_argument("--pause_seconds", type=float, default=0.0)
    live_parser.add_argument("--threshold", type=float, default=None)
    live_parser.add_argument("--file_max_threshold", type=float, default=0.5)
    live_parser.add_argument("--file_ratio_threshold", type=float, default=0.4)
    live_parser.add_argument("--min_records", type=int, default=1)
    live_parser.add_argument("--shap_sample_rows", type=int, default=50)
    live_parser.add_argument("--shap_top_n", type=int, default=15)
    live_parser.add_argument("--drift_n_bins", type=int, default=DEFAULT_DRIFT_N_BINS)
    live_parser.add_argument("--drift_min_category_count", type=int, default=DEFAULT_DRIFT_MIN_CATEGORY_COUNT)
    live_parser.add_argument("--numeric_threshold", type=float, default=0.95)
    live_parser.add_argument("--adwin_delta", type=float, default=DEFAULT_ADWIN_DELTA)
    live_parser.add_argument("--adwin_min_window", type=int, default=DEFAULT_ADWIN_MIN_WINDOW)
    live_parser.add_argument("--adwin_max_window", type=int, default=DEFAULT_ADWIN_MAX_WINDOW)
    live_parser.add_argument("--auto_retrain", action="store_true")
    live_parser.add_argument("--model_pointer_path", default=str(DEFAULT_MODEL_POINTER_PATH))
    live_parser.add_argument("--retrain_recent_rows", type=int, default=5000)
    live_parser.add_argument("--retrain_min_category_count", type=int, default=20)
    live_parser.add_argument("--tshark_timeout_seconds", type=int, default=120)
    live_parser.add_argument("--disable_live_drift", action="store_true")
    live_parser.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    live_parser.add_argument("--db_name", default="edge_iiot_paper")
    live_parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    live_parser.add_argument("--capture_dir", default=str(DEFAULT_CAPTURE_DIR))
    live_parser.add_argument("--injection_dir", default=str(DEFAULT_INJECTION_DIR))
    live_parser.add_argument("--processed_dir", default=str(DEFAULT_PROCESSED_DIR))
    live_parser.add_argument("--status_path", default=str(DEFAULT_STATUS_PATH))
    live_parser.add_argument("--no_metadata", action="store_true")
    live_parser.add_argument("--validate_tshark_fields", action="store_true")
    live_parser.set_defaults(func=record_loop)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "live":
        raise SystemExit("Only the live command is supported.")
    args.func(args)


if __name__ == "__main__":
    main()
