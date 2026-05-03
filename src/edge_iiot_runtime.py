from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_FEATURE_CONTRACT_PATH = DEFAULT_MODELS_DIR / "edge_iiot_feature_contract.json"
DEFAULT_ACTIVE_MODEL_POINTER_PATH = DEFAULT_MODELS_DIR / "edge_iiot_active_model.json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            converted = value.tolist()
            if converted is not value:
                return json_safe(converted)
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (datetime,)):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.isoformat()
    return value


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(json_safe(payload), fh, indent=2)
    return path


def feature_contract_from_meta(
    training_meta: dict[str, Any],
    *,
    dataset_name: str,
    model_family: str,
    label_source: str,
    version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = {
        "dataset_name": dataset_name,
        "model_family": model_family,
        "label_source": label_source,
        "version": version or utc_now().strftime("%Y%m%dT%H%M%SZ"),
        "created_at": utc_now(),
        "raw_edge_columns": list(training_meta.get("raw_edge_columns", [])),
        "raw_feature_columns_before_drops": list(training_meta.get("raw_feature_columns_before_drops", [])),
        "dropped_identity_payload_columns": list(training_meta.get("dropped_identity_payload_columns", [])),
        "dropped_empty_columns": list(training_meta.get("dropped_empty_columns", [])),
        "dropped_constant_columns": list(training_meta.get("dropped_constant_columns", [])),
        "feature_columns": list(training_meta.get("feature_columns", [])),
        "numeric_columns": list(training_meta.get("numeric_columns", [])),
        "categorical_columns": list(training_meta.get("categorical_columns", [])),
        "numeric_parse_ratios": dict(training_meta.get("numeric_parse_ratios", {})),
        "numeric_non_empty_counts": dict(training_meta.get("numeric_non_empty_counts", {})),
    }
    if extra:
        contract.update(extra)
    return contract


def save_feature_contract(
    training_meta: dict[str, Any],
    path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    *,
    dataset_name: str,
    model_family: str,
    label_source: str,
    version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    contract = feature_contract_from_meta(
        training_meta,
        dataset_name=dataset_name,
        model_family=model_family,
        label_source=label_source,
        version=version,
        extra=extra,
    )
    return write_json(path, contract)


def save_active_model_pointer(
    path: str | Path = DEFAULT_ACTIVE_MODEL_POINTER_PATH,
    *,
    binary_model_path: str | Path,
    attack_model_path: str | Path | None = None,
    feature_contract_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
    multiclass_threshold_path: str | Path | None = None,
    version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "kind": "active_model_pointer",
        "version": version or utc_now().strftime("%Y%m%dT%H%M%SZ"),
        "updated_at": utc_now(),
        "binary_model_path": str(binary_model_path),
        "attack_model_path": str(attack_model_path) if attack_model_path else None,
        "feature_contract_path": str(feature_contract_path) if feature_contract_path else None,
        "threshold_path": str(threshold_path) if threshold_path else None,
        "multiclass_threshold_path": str(multiclass_threshold_path) if multiclass_threshold_path else None,
    }
    if extra:
        payload.update(extra)
    return write_json(path, payload)


def load_active_model_pointer(path: str | Path = DEFAULT_ACTIVE_MODEL_POINTER_PATH) -> dict[str, Any] | None:
    pointer = read_json(path, default=None)
    if isinstance(pointer, dict) and pointer.get("kind") == "active_model_pointer":
        return pointer
    return None


def resolve_model_path(reference: str | Path, key: str = "binary_model_path") -> Path:
    ref = Path(reference)
    if ref.suffix.lower() != ".json":
        return ref
    pointer = load_active_model_pointer(ref)
    if not pointer:
        raise FileNotFoundError(f"Active model pointer not found or invalid: {ref}")
    model_path = pointer.get(key)
    if not model_path:
        raise FileNotFoundError(f"Active model pointer {ref} does not contain {key}")
    return Path(model_path)


def transformed_feature_names_from_bundle(bundle: dict[str, Any]) -> list[str]:
    names = bundle.get("transformed_feature_names")
    if names:
        return [str(name) for name in names]

    preprocessor = bundle.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            pass

    training_meta = bundle.get("training_meta", {})
    return [str(name) for name in training_meta.get("feature_columns", [])]


def attach_model_feature_names(bundle: dict[str, Any], feature_names: list[str] | None = None) -> list[str]:
    names = feature_names or transformed_feature_names_from_bundle(bundle)
    model = bundle.get("model")
    if model is not None and names:
        try:
            setattr(model, "feature_names_", list(names))
        except Exception:
            pass
    bundle["transformed_feature_names"] = list(names)
    return list(names)


def validate_runtime_contract(
    *,
    bundle: dict[str, Any],
    model_input: pd.DataFrame,
    transformed: Any | None = None,
    transformed_feature_names: list[str] | None = None,
    stage: str = "inference",
) -> dict[str, Any]:
    training_meta = bundle.get("training_meta", {})
    feature_columns = [str(col) for col in training_meta.get("feature_columns", [])]
    numeric_columns = [str(col) for col in training_meta.get("numeric_columns", [])]
    categorical_columns = [str(col) for col in training_meta.get("categorical_columns", [])]

    if not feature_columns:
        raise ValueError(f"{stage}: model bundle is missing training_meta.feature_columns.")

    input_columns = [str(col) for col in model_input.columns]
    if input_columns != feature_columns:
        missing = [col for col in feature_columns if col not in input_columns]
        extra = [col for col in input_columns if col not in feature_columns]
        raise ValueError(
            f"{stage}: feature contract mismatch. Expected {len(feature_columns)} columns in training order. "
            f"Missing={missing[:10]} Extra={extra[:10]}."
        )

    non_numeric = [col for col in numeric_columns if col in model_input.columns and not pd.api.types.is_numeric_dtype(model_input[col])]
    if non_numeric:
        raise TypeError(f"{stage}: numeric feature columns are not numeric after preprocessing: {non_numeric[:10]}.")

    missing_categorical = [col for col in categorical_columns if col not in model_input.columns]
    if missing_categorical:
        raise ValueError(f"{stage}: categorical feature columns are missing: {missing_categorical[:10]}.")

    names = attach_model_feature_names(bundle, transformed_feature_names)
    model = bundle.get("model")
    if transformed is not None:
        transformed_columns = int(getattr(transformed, "shape", (0, 0))[1])
        if names and transformed_columns != len(names):
            raise ValueError(
                f"{stage}: transformed vector length {transformed_columns} does not match "
                f"{len(names)} transformed feature names."
            )
        model_n_features = getattr(model, "n_features_in_", None) if model is not None else None
        if model_n_features is not None and int(model_n_features) != transformed_columns:
            raise ValueError(
                f"{stage}: transformed vector length {transformed_columns} does not match "
                f"model.n_features_in_={model_n_features}."
            )

    model_feature_names = list(getattr(model, "feature_names_", names) or []) if model is not None else names
    if names and model_feature_names and [str(name) for name in model_feature_names] != names:
        raise ValueError(f"{stage}: model.feature_names_ does not match transformed feature names.")

    return {
        "stage": stage,
        "feature_count": len(feature_columns),
        "numeric_count": len(numeric_columns),
        "categorical_count": len(categorical_columns),
        "transformed_feature_count": len(names),
    }
