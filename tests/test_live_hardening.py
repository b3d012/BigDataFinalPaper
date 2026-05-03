from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edge_iiot_dashboard import clear_directory_files, preflight_live_capture
from edge_iiot_live_capture import prediction_documents, update_adwin_monitor


def test_clear_directory_files_skips_locked_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        locked = folder / "live_window_000001.csv"
        unlocked = folder / "live_window_000002.csv"
        locked.write_text("locked", encoding="utf-8")
        unlocked.write_text("unlocked", encoding="utf-8")

        original_unlink = Path.unlink
        def fake_unlink(self: Path, *args, **kwargs):
            if self == locked:
                raise PermissionError("locked")
            return original_unlink(self, *args, **kwargs)

        with mock.patch("pathlib.Path.unlink", new=fake_unlink):
            result = clear_directory_files(folder)

        assert result["deleted"] == 1
        assert result["skipped"] == [str(locked)]


def test_clear_directory_files_recovers_after_transient_lock() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        locked = folder / "live_window_000001.csv"
        locked.write_text("locked", encoding="utf-8")

        original_unlink = Path.unlink
        seen_locked = {"count": 0}

        def fake_unlink(self: Path, *args, **kwargs):
            if self == locked and seen_locked["count"] == 0:
                seen_locked["count"] += 1
                raise PermissionError("locked")
            return original_unlink(self, *args, **kwargs)

        with mock.patch("pathlib.Path.unlink", new=fake_unlink):
            result = clear_directory_files(folder)

        assert result["deleted"] == 1
        assert result["skipped"] == []


def test_preflight_live_capture_rejects_invalid_interface() -> None:
    with mock.patch("edge_iiot_dashboard.find_tshark", return_value=r"C:\Program Files\Wireshark\tshark.exe"), \
         mock.patch("edge_iiot_dashboard.validate_tshark_interface", return_value=(False, "Interface 'bogus' is not available.")):
        ok, message, resolved = preflight_live_capture("bogus", "tshark")

    assert ok is False
    assert "bogus" in message
    assert resolved is not None


def test_preflight_live_capture_accepts_valid_interface() -> None:
    with mock.patch("edge_iiot_dashboard.find_tshark", return_value=r"C:\Program Files\Wireshark\tshark.exe"), \
         mock.patch("edge_iiot_dashboard.validate_tshark_interface", return_value=(True, "")):
        ok, message, resolved = preflight_live_capture("Ethernet", "tshark")

    assert ok is True
    assert message == ""
    assert resolved == r"C:\Program Files\Wireshark\tshark.exe"


def test_prediction_documents_handles_attack_bundle() -> None:
    raw_df = pd.DataFrame([{"tcp.flags": 1, "mqtt.len": 0}])
    docs = prediction_documents(
        window_id=1,
        interface="Ethernet",
        raw_df=raw_df,
        proba=np.array([0.987]),
        pred=np.array([1]),
        attack_proba=np.array([0.999]),
        attack_pred=np.array([2]),
        threshold=0.5,
        raw_csv=Path("dummy.csv"),
        source_file_tag="capture.pcap",
        capture_mode="live_capture",
        model_version="v1",
        attack_model_version="v2",
        attack_bundle={"classes_": ["Normal", "DDoS", "Ransomware"]},
        true_label=1,
    )

    assert len(docs) == 1
    assert docs[0]["attack_type_name"] == "Ransomware"
    assert docs[0]["pred_proba_attack"] == 0.987


def test_prediction_documents_override_benign_anomaly_and_store_top3_shap() -> None:
    raw_df = pd.DataFrame([{"tcp.flags": 0, "mqtt.len": 10}])
    shap = {
        0: [
            {"feature": "a", "contribution": 1.0, "abs_contribution": 1.0, "rank": 1},
            {"feature": "b", "contribution": 0.8, "abs_contribution": 0.8, "rank": 2},
            {"feature": "c", "contribution": 0.4, "abs_contribution": 0.4, "rank": 3},
            {"feature": "d", "contribution": 0.2, "abs_contribution": 0.2, "rank": 4},
        ]
    }

    docs = prediction_documents(
        window_id=1,
        interface="Ethernet",
        raw_df=raw_df,
        proba=np.array([0.1]),
        pred=np.array([0]),
        attack_proba=None,
        attack_pred=None,
        threshold=0.5,
        raw_csv=Path("dummy.csv"),
        source_file_tag="capture.pcap",
        capture_mode="live_capture",
        model_version="v1",
        attack_model_version="v1",
        true_label=0,
        shap_by_record=shap,
        drift_status={"delta": 0.002},
        anomaly_scores=np.array([0.9]),
        anomaly_pred=np.array([1]),
        scoring_latency_ms=12.5,
        shap_latency_ms=4.0,
    )

    doc = docs[0]
    assert doc["status"] == "Potential Evasion/Anomalous"
    assert doc["alert_flag"] is True
    assert doc["anomaly_override"] is True
    assert len(doc["shap_payload"]["top_features"]) == 3
    assert doc["alert_latency_ms"] == 12.5
    assert doc["scoring_latency_ms"] == 12.5
    assert doc["micro_batch_latency_ms"] == 12.5
    assert doc["shap_latency_ms"] == 4.0
    assert doc["shap_payload"]["shap_latency_ms"] == 4.0
    assert doc["drift_status"]["delta"] == 0.002
    assert doc["flow_data"]["tcp.flags"] == 0
    assert doc["prediction"]["status"] == "Potential Evasion/Anomalous"


def test_update_adwin_monitor_records_drift_latency() -> None:
    class DriftOnce:
        def __init__(self) -> None:
            self.seen = 0

        def update(self, value: float):
            self.seen += 1
            return {
                "drift_detected": self.seen == 1,
                "cut_index": 1,
                "mean_before": 0.9,
                "mean_after": 0.5,
                "window_width": 2,
                "running_mean": value,
                "running_std": 0.0,
                "drift_count": 1,
                "last_drift_index": self.seen,
                "delta": 0.002,
            }

    with mock.patch("edge_iiot_live_capture.insert_documents") as insert:
        _, events = update_adwin_monitor(
            adwin=DriftOnce(),
            scores=np.array([0.99]),
            db={"drift_events": object()},
            window_id=1,
            interface="Ethernet",
            source_file_tag="window",
            capture_mode="live_capture",
            batch_started_perf=0.0,
        )

    assert insert.called
    assert events
    assert "drift_detection_latency_ms" in events[-1]
