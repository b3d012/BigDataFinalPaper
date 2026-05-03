from __future__ import annotations

import json
from pathlib import Path


def test_mongo_schema_lists_required_collections() -> None:
    schema_path = Path("schemas/edge_iiot_mongo_schema.json")
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    collections = data["collections"]
    required = {
        "predictions",
        "alert_explanations",
        "drift_events",
        "retrain_events",
        "adversarial_evaluations",
    }
    assert required.issubset(collections.keys())
