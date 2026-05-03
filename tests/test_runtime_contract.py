from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edge_iiot_runtime import load_active_model_pointer, read_json, save_active_model_pointer, save_feature_contract


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
