from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edge_iiot_adwin import ADWINMonitor


def test_adwin_detects_mean_shift() -> None:
    monitor = ADWINMonitor(delta=0.2, min_window=5, max_window=200)
    drift_detected = False
    for value in [0.05] * 30 + [0.95] * 30:
        state = monitor.update(value)
        drift_detected = drift_detected or bool(state["drift_detected"])

    assert drift_detected or monitor.state.drift_count > 0
    assert monitor.state.total_seen == 60
