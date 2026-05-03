from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edge_iiot_runtime import load_active_model_pointer, read_json, save_active_model_pointer, save_feature_contract
from edge_iiot_runtime import attach_model_feature_names, validate_runtime_contract

import numpy as np
import pandas as pd
import pytest


def test_feature_contract_and_pointer_round_trip(tmp_path: Path) -> None:
    training_meta = {
        "raw_edge_columns": ["a", "b"],
        "raw_feature_columns_before_drops": ["a", "b"],
        "dropped_identity_payload_columns": [],
        "dropped_empty_columns": [],
        "dropped_constant_columns": [],
        "feature_columns": ["a", "b"],
        "numeric_columns": ["a"],
        "categorical_columns": ["b"],
        "numeric_parse_ratios": {"a": 1.0},
        "numeric_non_empty_counts": {"a": 2},
    }

    contract_path = tmp_path / "feature_contract.json"
    pointer_path = tmp_path / "active_model.json"

    saved_contract = save_feature_contract(
        training_meta,
        contract_path,
        dataset_name="edge",
        model_family="binary_xgb",
        label_source="Attack_label",
        version="v1",
        extra={"threshold": 0.5},
    )
    saved_pointer = save_active_model_pointer(
        pointer_path,
        binary_model_path="models/model.joblib",
        feature_contract_path=saved_contract,
        threshold_path="models/model.metadata.json",
        version="v1",
        extra={"kind": "active_model_pointer"},
    )

    contract = read_json(saved_contract)
    pointer = load_active_model_pointer(saved_pointer)

    assert contract["dataset_name"] == "edge"
    assert contract["feature_columns"] == ["a", "b"]
    assert pointer is not None
    assert pointer["binary_model_path"].endswith("model.joblib")


class DummyModel:
    def __init__(self, n_features: int) -> None:
        self.n_features_in_ = n_features


def test_runtime_contract_accepts_matching_frame() -> None:
    bundle = {
        "model": DummyModel(2),
        "training_meta": {
            "feature_columns": ["a", "b"],
            "numeric_columns": ["a", "b"],
            "categorical_columns": [],
        },
    }
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    transformed = np.array([[1.0, 2.0]])

    result = validate_runtime_contract(
        bundle=bundle,
        model_input=frame,
        transformed=transformed,
        transformed_feature_names=["a", "b"],
        stage="test",
    )

    assert result["feature_count"] == 2
    assert bundle["model"].feature_names_ == ["a", "b"]


def test_runtime_contract_rejects_misordered_features() -> None:
    bundle = {
        "model": DummyModel(2),
        "training_meta": {
            "feature_columns": ["a", "b"],
            "numeric_columns": ["a", "b"],
            "categorical_columns": [],
        },
    }
    frame = pd.DataFrame({"b": [2.0], "a": [1.0]})

    with pytest.raises(ValueError, match="feature contract mismatch"):
        validate_runtime_contract(bundle=bundle, model_input=frame, stage="test")


def test_runtime_contract_rejects_dtype_mismatch() -> None:
    bundle = {
        "model": DummyModel(2),
        "training_meta": {
            "feature_columns": ["a", "b"],
            "numeric_columns": ["a", "b"],
            "categorical_columns": [],
        },
    }
    frame = pd.DataFrame({"a": ["bad"], "b": [2.0]})

    with pytest.raises(TypeError, match="not numeric"):
        validate_runtime_contract(bundle=bundle, model_input=frame, stage="test")


def test_runtime_contract_rejects_model_feature_count_mismatch() -> None:
    bundle = {
        "model": DummyModel(3),
        "training_meta": {
            "feature_columns": ["a", "b"],
            "numeric_columns": ["a", "b"],
            "categorical_columns": [],
        },
    }
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})

    with pytest.raises(ValueError, match="model.n_features_in_"):
        validate_runtime_contract(
            bundle=bundle,
            model_input=frame,
            transformed=np.array([[1.0, 2.0]]),
            transformed_feature_names=["a", "b"],
            stage="test",
        )


def test_attach_model_feature_names_sets_bundle_and_model() -> None:
    bundle = {
        "model": DummyModel(2),
        "training_meta": {"feature_columns": ["a", "b"]},
    }

    names = attach_model_feature_names(bundle, ["a", "b"])

    assert names == ["a", "b"]
    assert bundle["transformed_feature_names"] == ["a", "b"]
    assert bundle["model"].feature_names_ == ["a", "b"]
