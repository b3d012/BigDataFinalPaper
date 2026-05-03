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
from edge_iiot_live_capture import prediction_documents


def test_clear_directory_files_skips_locked_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        locked = folder / "live_window_000001.csv"
        unlocked = folder / "live_window_000002.csv"
        locked.write_text("locked", encoding="utf-8")
        unlocked.write_text("unlocked", encoding="utf-8")

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
