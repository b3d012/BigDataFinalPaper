from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edge_iiot_multiclass import calibrate_multiclass_thresholds


def test_multiclass_thresholds_include_critical_class() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    proba = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=float,
    )

    result = calibrate_multiclass_thresholds(
        y_true,
        proba,
        classes=["Benign", "Ransomware"],
        critical_classes=["Ransomware"],
        min_precision=0.5,
    )

    assert len(result["per_class"]) == 2
    table = result["table"]
    assert not table.empty
    critical_row = next(row for row in result["per_class"] if row["class_name"] == "Ransomware")
    assert critical_row["critical"] is True
    assert critical_row["strategy"] == "f2"
