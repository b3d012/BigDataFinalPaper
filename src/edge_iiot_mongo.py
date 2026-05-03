from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pymongo import ASCENDING, MongoClient, ReplaceOne
from pymongo.errors import PyMongoError


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DEFAULT_MONGO_DB = os.getenv("MONGODB_DB", "edge_iiot_paper")

RAW_PACKETS_COLLECTION = "raw_packets"
FEATURE_VECTORS_COLLECTION = "feature_vectors"
PREDICTIONS_COLLECTION = "predictions"
ALERT_EXPLANATIONS_COLLECTION = "alert_explanations"
LIVE_WINDOWS_COLLECTION = "live_windows"
FEATURE_IMPORTANCE_COLLECTION = "feature_importance"
THRESHOLD_METRICS_COLLECTION = "threshold_metrics"
DRIFT_EVENTS_COLLECTION = "drift_events"
RETRAIN_EVENTS_COLLECTION = "retrain_events"
ADVERSARIAL_EVALUATIONS_COLLECTION = "adversarial_evaluations"
ANALYSIS_SUMMARIES_COLLECTION = "analysis_summaries"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        if pd.isna(value):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def sanitize_record(record: dict[str, object]) -> dict[str, object]:
    return {key: to_builtin(value) for key, value in record.items()}


def connect_database(uri: str, db_name: str):
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[db_name]


def ensure_indexes(db) -> None:
    db[RAW_PACKETS_COLLECTION].create_index([("kind", ASCENDING), ("source_file", ASCENDING), ("record_index", ASCENDING)])
    db[FEATURE_VECTORS_COLLECTION].create_index([("kind", ASCENDING), ("source_file", ASCENDING), ("record_index", ASCENDING)])
    db[PREDICTIONS_COLLECTION].create_index([("kind", ASCENDING), ("source", ASCENDING), ("source_file", ASCENDING), ("record_index", ASCENDING)])
    db[PREDICTIONS_COLLECTION].create_index([("kind", ASCENDING), ("source", ASCENDING), ("created_at", ASCENDING)])
    db[ALERT_EXPLANATIONS_COLLECTION].create_index([("kind", ASCENDING), ("source", ASCENDING), ("source_file", ASCENDING), ("record_index", ASCENDING)])
    db[LIVE_WINDOWS_COLLECTION].create_index([("window_id", ASCENDING), ("source", ASCENDING), ("created_at", ASCENDING)])
    db[FEATURE_IMPORTANCE_COLLECTION].create_index([("importance_type", ASCENDING), ("feature", ASCENDING)])
    db[THRESHOLD_METRICS_COLLECTION].create_index([("kind", ASCENDING), ("source", ASCENDING), ("threshold", ASCENDING)])
    db[DRIFT_EVENTS_COLLECTION].create_index([("kind", ASCENDING), ("source", ASCENDING), ("window_id", ASCENDING), ("created_at", ASCENDING)])
    db[RETRAIN_EVENTS_COLLECTION].create_index([("kind", ASCENDING), ("source", ASCENDING), ("created_at", ASCENDING)])
    db[ADVERSARIAL_EVALUATIONS_COLLECTION].create_index([("kind", ASCENDING), ("source", ASCENDING), ("created_at", ASCENDING)])
    db[ANALYSIS_SUMMARIES_COLLECTION].create_index([("kind", ASCENDING), ("artifact_path", ASCENDING)])
    db[ANALYSIS_SUMMARIES_COLLECTION].create_index([("kind", ASCENDING), ("created_at", ASCENDING)])


def upsert_documents(collection, documents: Iterable[dict[str, object]], key_fields: list[str]) -> int:
    inserted = 0
    batch: list[ReplaceOne] = []
    for document in documents:
        document = sanitize_record(document)
        filter_query = {field: document.get(field) for field in key_fields}
        batch.append(ReplaceOne(filter_query, document, upsert=True))
        if len(batch) >= 1000:
            collection.bulk_write(batch, ordered=False)
            inserted += len(batch)
            batch = []
    if batch:
        collection.bulk_write(batch, ordered=False)
        inserted += len(batch)
    return inserted


def insert_documents(collection, documents: Iterable[dict[str, object]]) -> int:
    records = [sanitize_record(document) for document in documents]
    if not records:
        return 0
    collection.insert_many(records, ordered=False)
    return len(records)


def dataframe_to_documents(
    df: pd.DataFrame,
    *,
    kind: str,
    source: str,
    source_file: str | None = None,
    extra_fields: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    extra_fields = extra_fields or {}
    for record_index, row in df.reset_index(drop=True).iterrows():
        document = {
            "kind": kind,
            "source": source,
            "source_file": source_file,
            "record_index": int(record_index),
            "created_at": utc_now(),
            **extra_fields,
            **row.to_dict(),
        }
        documents.append(document)
    return documents


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def seed_prediction_csv(
    db,
    path: Path,
    *,
    source: str,
    kind: str = "classifier_prediction",
    threshold: float | None = None,
    extra_fields: dict[str, object] | None = None,
) -> int:
    df = read_csv(path)
    records = []
    for record_index, row in df.reset_index(drop=True).iterrows():
        payload = {
            "kind": kind,
            "source": source,
            "source_file": path.name,
            "record_index": int(record_index),
            "created_at": utc_now(),
            "threshold": threshold,
            **(extra_fields or {}),
            **row.to_dict(),
        }
        if "correct" not in payload and {"true_label", "pred_label"}.issubset(payload):
            payload["correct"] = int(payload["true_label"] == payload["pred_label"])
        records.append(payload)
    return upsert_documents(db[PREDICTIONS_COLLECTION], records, ["kind", "source", "source_file", "record_index"])


def seed_feature_importance_csv(
    db,
    path: Path,
    *,
    importance_type: str,
    source: str,
    extra_fields: dict[str, object] | None = None,
) -> int:
    df = read_csv(path)
    records = []
    for _, row in df.iterrows():
        payload = {
            "importance_type": importance_type,
            "source": source,
            "artifact_path": str(path),
            "created_at": utc_now(),
            **(extra_fields or {}),
            **row.to_dict(),
        }
        records.append(payload)
    return upsert_documents(db[FEATURE_IMPORTANCE_COLLECTION], records, ["importance_type", "source", "feature"])


def seed_threshold_grid(db, path: Path, *, source: str, kind: str) -> int:
    df = read_csv(path)
    records = []
    for _, row in df.iterrows():
        payload = {
            "kind": kind,
            "source": source,
            "artifact_path": str(path),
            "created_at": utc_now(),
            **row.to_dict(),
        }
        records.append(payload)
    return upsert_documents(db[THRESHOLD_METRICS_COLLECTION], records, ["kind", "source", "threshold"])


def seed_summary_document(db, path: Path, *, kind: str, source: str = "offline") -> int:
    if path.suffix.lower() == ".json":
        payload = read_json(path)
    else:
        payload = {"text": read_text(path)}
    document = {
        "kind": kind,
        "source": source,
        "artifact_path": str(path),
        "created_at": utc_now(),
        "payload": payload,
    }
    upsert_documents(db[ANALYSIS_SUMMARIES_COLLECTION], [document], ["kind", "artifact_path"])
    return 1


def seed_demo_extracted_csvs(db, folder: Path) -> dict[str, int]:
    if not folder.exists():
        return {"raw_packets": 0, "feature_vectors": 0}
    counts = {"raw_packets": 0, "feature_vectors": 0}
    for csv_path in sorted(folder.glob("*.csv")):
        df = read_csv(csv_path)
        raw_docs = dataframe_to_documents(df, kind="demo_raw_packet", source="demo", source_file=csv_path.name)
        feature_docs = dataframe_to_documents(df, kind="demo_feature_vector", source="demo", source_file=csv_path.name)
        counts["raw_packets"] += upsert_documents(db[RAW_PACKETS_COLLECTION], raw_docs, ["kind", "source_file", "record_index"])
        counts["feature_vectors"] += upsert_documents(db[FEATURE_VECTORS_COLLECTION], feature_docs, ["kind", "source_file", "record_index"])
    return counts


def seed_offline_artifacts(db, repo_root: Path = REPO_ROOT) -> dict[str, int]:
    report_dir = repo_root / "output" / "reports"
    demo_dir = repo_root / "output" / "demo"
    demo_counts = seed_demo_extracted_csvs(db, demo_dir / "extracted_csvs")
    counts = {
        "raw_packets": demo_counts["raw_packets"],
        "feature_vectors": demo_counts["feature_vectors"],
        "predictions": 0,
        "alert_explanations": 0,
        "feature_importance": 0,
        "threshold_metrics": 0,
        "drift_events": 0,
        "retrain_events": 0,
        "adversarial_evaluations": 0,
        "analysis_summaries": 0,
    }

    prediction_specs = [
        (report_dir / "edge_iiot_holdout_predictions.csv", "holdout", "classifier_prediction", None),
        (report_dir / "edge_iiot_cv_predictions.csv", "cv", "classifier_prediction", None),
        (demo_dir / "edge_iiot_demo_predictions.csv", "demo", "classifier_prediction", None),
        (report_dir / "edge_iiot_anomaly_holdout_predictions.csv", "holdout", "anomaly_prediction", None),
        (demo_dir / "edge_iiot_demo_anomaly_predictions.csv", "demo", "anomaly_prediction", None),
    ]
    for path, source, kind, threshold in prediction_specs:
        if path.exists():
            counts["predictions"] += seed_prediction_csv(db, path, source=source, kind=kind, threshold=threshold)

    alert_specs = [
        (report_dir / "edge_iiot_shap_local_examples.csv", "shap_local", "classifier"),
    ]
    for path, kind, source in alert_specs:
        if path.exists():
            counts["alert_explanations"] += seed_feature_importance_csv(db, path, importance_type=kind, source=source)

    feature_specs = [
        (repo_root / "models" / "edge_iiot_xgb_model.feature_importance.csv", "model_feature_importance", "classifier"),
        (report_dir / "edge_iiot_shap_global_importance.csv", "shap", "classifier"),
        (report_dir / "edge_iiot_drift_feature_scores.csv", "drift", "drift"),
    ]
    for path, importance_type, source in feature_specs:
        if path.exists():
            counts["feature_importance"] += seed_feature_importance_csv(db, path, importance_type=importance_type, source=source)

    threshold_specs = [
        (report_dir / "edge_iiot_holdout_threshold_grid.csv", "holdout", "threshold_grid"),
        (report_dir / "edge_iiot_cv_threshold_grid.csv", "cv", "threshold_grid"),
    ]
    for path, source, kind in threshold_specs:
        if path.exists():
            counts["threshold_metrics"] += seed_threshold_grid(db, path, source=source, kind=kind)

    summary_specs = [
        (report_dir / "edge_iiot_holdout_metrics.json", "holdout_metrics"),
        (report_dir / "edge_iiot_cv_summary.json", "cv_summary"),
        (report_dir / "edge_iiot_multiclass_summary.json", "multiclass_summary"),
        (report_dir / "edge_iiot_shap_summary.json", "shap_summary"),
        (report_dir / "edge_iiot_anomaly_holdout_metrics.json", "anomaly_holdout_metrics"),
        (report_dir / "edge_iiot_drift_summary.json", "drift_summary"),
        (report_dir / "edge_iiot_retrain_trigger.json", "retrain_trigger"),
        (report_dir / "edge_iiot_retrain_comparison.json", "retrain_comparison"),
        (report_dir / "edge_iiot_adversarial_robustness.json", "adversarial_robustness"),
        (demo_dir / "edge_iiot_demo_drift_summary.json", "demo_drift_summary"),
        (repo_root / "models" / "edge_iiot_feature_contract.json", "feature_contract"),
        (repo_root / "models" / "edge_iiot_active_model.json", "active_model_pointer"),
    ]
    for path, kind in summary_specs:
        if path.exists():
            counts["analysis_summaries"] += seed_summary_document(db, path, kind=kind)

    text_specs = [
        (report_dir / "edge_iiot_threshold_calibration_summary.md", "threshold_calibration_summary"),
        (report_dir / "edge_iiot_anomaly_run_summary.md", "anomaly_run_summary"),
        (report_dir / "edge_iiot_drift_run_summary.md", "drift_run_summary"),
        (report_dir / "edge_iiot_retrain_run_summary.md", "retrain_run_summary"),
        (demo_dir / "edge_iiot_demo_run_summary.md", "demo_run_summary"),
    ]
    for path, kind in text_specs:
        if path.exists():
            counts["analysis_summaries"] += seed_summary_document(db, path, kind=kind)

    adversarial_path = report_dir / "edge_iiot_adversarial_robustness.json"
    if adversarial_path.exists():
        payload = read_json(adversarial_path)
        record = {
            "kind": "adversarial_robustness",
            "source": "offline",
            "artifact_path": str(adversarial_path),
            "created_at": utc_now(),
            "payload": payload,
        }
        counts["adversarial_evaluations"] += upsert_documents(
            db[ADVERSARIAL_EVALUATIONS_COLLECTION],
            [record],
            ["kind", "artifact_path"],
        )

    return counts


def collection_counts(db) -> dict[str, int]:
    return {
        RAW_PACKETS_COLLECTION: db[RAW_PACKETS_COLLECTION].count_documents({}),
        FEATURE_VECTORS_COLLECTION: db[FEATURE_VECTORS_COLLECTION].count_documents({}),
        PREDICTIONS_COLLECTION: db[PREDICTIONS_COLLECTION].count_documents({}),
        ALERT_EXPLANATIONS_COLLECTION: db[ALERT_EXPLANATIONS_COLLECTION].count_documents({}),
        LIVE_WINDOWS_COLLECTION: db[LIVE_WINDOWS_COLLECTION].count_documents({}),
        FEATURE_IMPORTANCE_COLLECTION: db[FEATURE_IMPORTANCE_COLLECTION].count_documents({}),
        THRESHOLD_METRICS_COLLECTION: db[THRESHOLD_METRICS_COLLECTION].count_documents({}),
        DRIFT_EVENTS_COLLECTION: db[DRIFT_EVENTS_COLLECTION].count_documents({}),
        RETRAIN_EVENTS_COLLECTION: db[RETRAIN_EVENTS_COLLECTION].count_documents({}),
        ADVERSARIAL_EVALUATIONS_COLLECTION: db[ADVERSARIAL_EVALUATIONS_COLLECTION].count_documents({}),
        ANALYSIS_SUMMARIES_COLLECTION: db[ANALYSIS_SUMMARIES_COLLECTION].count_documents({}),
    }


def clear_live_collections(db) -> dict[str, int]:
    """Remove live-only records while preserving offline artifacts."""
    live_query = {"kind": {"$regex": "^live_"}}
    deleted = {
        RAW_PACKETS_COLLECTION: db[RAW_PACKETS_COLLECTION].delete_many(live_query).deleted_count,
        FEATURE_VECTORS_COLLECTION: db[FEATURE_VECTORS_COLLECTION].delete_many(live_query).deleted_count,
        PREDICTIONS_COLLECTION: db[PREDICTIONS_COLLECTION].delete_many(live_query).deleted_count,
        ALERT_EXPLANATIONS_COLLECTION: db[ALERT_EXPLANATIONS_COLLECTION].delete_many(live_query).deleted_count,
        LIVE_WINDOWS_COLLECTION: db[LIVE_WINDOWS_COLLECTION].delete_many(live_query).deleted_count,
        FEATURE_IMPORTANCE_COLLECTION: db[FEATURE_IMPORTANCE_COLLECTION].delete_many(live_query).deleted_count,
        THRESHOLD_METRICS_COLLECTION: db[THRESHOLD_METRICS_COLLECTION].delete_many({"source": "live"}).deleted_count,
        DRIFT_EVENTS_COLLECTION: db[DRIFT_EVENTS_COLLECTION].delete_many(live_query).deleted_count,
        RETRAIN_EVENTS_COLLECTION: db[RETRAIN_EVENTS_COLLECTION].delete_many(live_query).deleted_count,
        ADVERSARIAL_EVALUATIONS_COLLECTION: db[ADVERSARIAL_EVALUATIONS_COLLECTION].delete_many(live_query).deleted_count,
        ANALYSIS_SUMMARIES_COLLECTION: db[ANALYSIS_SUMMARIES_COLLECTION].delete_many(live_query).deleted_count,
        "analysis_summaries_live_thresholds": db[ANALYSIS_SUMMARIES_COLLECTION].delete_many({"source": "live"}).deleted_count,
    }
    return deleted


def collection_to_dataframe(
    collection,
    *,
    query: dict[str, object] | None = None,
    projection: dict[str, int] | None = None,
    sort: list[tuple[str, int]] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    cursor = collection.find(query or {}, projection)
    if sort:
        cursor = cursor.sort(sort)
    if limit is not None:
        cursor = cursor.limit(limit)
    records = list(cursor)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for column in df.columns:
        if df[column].map(lambda value: isinstance(value, datetime)).any():
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    return df


def latest_summary_document(db, kind: str) -> dict[str, object] | None:
    document = db[ANALYSIS_SUMMARIES_COLLECTION].find_one({"kind": kind}, sort=[("created_at", -1)], projection={"_id": 0})
    if document is None:
        return None
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MongoDB helper for the Edge-IIoT paper pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed", help="Seed MongoDB with offline artifacts.")
    seed_parser.add_argument("--mongo_uri", default=DEFAULT_MONGO_URI)
    seed_parser.add_argument("--db_name", default=DEFAULT_MONGO_DB)

    status_parser = subparsers.add_parser("status", help="Check MongoDB connectivity and collection counts.")
    status_parser.add_argument("--mongo_uri", default=DEFAULT_MONGO_URI)
    status_parser.add_argument("--db_name", default=DEFAULT_MONGO_DB)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        db = connect_database(args.mongo_uri, args.db_name)
        ensure_indexes(db)
    except PyMongoError as exc:
        raise SystemExit(f"MongoDB connection failed: {exc}") from exc

    if args.command == "seed":
        counts = seed_offline_artifacts(db)
        print("MongoDB seeding complete.")
        for name, count in counts.items():
            print(f"{name}: {count}")
        return

    if args.command == "status":
        print("MongoDB connection OK.")
        print(f"Database: {args.db_name}")
        for name, count in collection_counts(db).items():
            print(f"{name}: {count}")
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
