from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from edge_iiot_experiment import coerce_feature_types, get_transformed_feature_names, normalize_columns
from edge_iiot_runtime import attach_model_feature_names, validate_runtime_contract


DEFAULT_DEMO_DIR = Path("demo")
DEFAULT_REFERENCE_SUMMARY = Path("demo/edge_iiot_attack_predictions_summary.csv")
DEFAULT_OUTPUT_DIR = Path("output/demo")
DEFAULT_EXTRACTED_DIR = DEFAULT_OUTPUT_DIR / "extracted_csvs"
DEFAULT_MODEL_PATH = Path("models/edge_iiot_xgb_model.joblib")
DEFAULT_PREDICTIONS_CSV = DEFAULT_OUTPUT_DIR / "edge_iiot_demo_predictions.csv"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "edge_iiot_demo_predictions_summary.csv"
DEFAULT_COMPARISON_CSV = DEFAULT_OUTPUT_DIR / "edge_iiot_demo_vs_reference_comparison.csv"
DEFAULT_RUN_SUMMARY_MD = DEFAULT_OUTPUT_DIR / "edge_iiot_demo_run_summary.md"
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
DEFAULT_FILE_MAX_THRESHOLD = 0.5
DEFAULT_FILE_RATIO_THRESHOLD = 0.4


def find_tshark(explicit_path: str | None = None) -> str:
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)

    found = shutil.which("tshark")
    if found:
        candidates.append(found)

    system = platform.system().lower()
    if system == "windows":
        candidates.extend(
            [
                r"C:\Program Files\Wireshark\tshark.exe",
                r"C:\Program Files (x86)\Wireshark\tshark.exe",
            ]
        )
    elif system == "darwin":
        candidates.append("/Applications/Wireshark.app/Contents/MacOS/tshark")

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))

    raise RuntimeError(
        "Could not find tshark. Install Wireshark CLI tools and make sure `tshark` is on PATH, "
        "or pass --tshark with the full path."
    )


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def available_tshark_fields(tshark: str) -> set[str] | None:
    try:
        output = run_text([tshark, "-G", "fields"])
    except Exception:
        return None

    fields = set()
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == "F":
            fields.add(parts[2])
    return fields


def available_tshark_interfaces(tshark: str) -> list[str]:
    try:
        output = run_text([tshark, "-D"])
    except Exception:
        return []
    interfaces: list[str] = []
    for line in output.splitlines():
        cleaned = line.strip()
        if cleaned:
            interfaces.append(cleaned)
    return interfaces


def validate_tshark_interface(tshark: str, interface: str) -> tuple[bool, str]:
    candidate = str(interface).strip()
    if not candidate:
        return False, "Interface is empty."

    interfaces = available_tshark_interfaces(tshark)
    if not interfaces:
        return False, "Unable to enumerate tshark interfaces."

    if candidate.isdigit():
        idx = int(candidate)
        if 1 <= idx <= len(interfaces):
            return True, ""

    normalized = candidate.lower()
    for line in interfaces:
        line_lower = line.lower()
        if normalized == line_lower or normalized in line_lower:
            return True, ""
        if ". " in line:
            label = line.split(". ", 1)[1].strip()
            label_lower = label.lower()
            if normalized == label_lower or normalized in label_lower:
                return True, ""

    preview = "; ".join(interfaces[:8])
    if len(interfaces) > 8:
        preview += "; ..."
    return False, f"Interface '{candidate}' is not available. Available interfaces: {preview}"


def pcap_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Demo folder not found: {folder}")
    files = sorted(
        [
            *folder.glob("*.pcap"),
            *folder.glob("*.pcapng"),
            *folder.glob("*.cap"),
        ]
    )
    if not files:
        raise RuntimeError(f"No PCAP/PCAPNG/CAP files found in {folder}")
    return files


def normalize_filename(name: str) -> str:
    return Path(str(name).strip()).stem.strip().lower()


def bundle_field_contract(bundle: dict[str, object], include_metadata: bool) -> list[str]:
    meta = bundle["training_meta"]
    fields = list(meta["feature_columns"])
    if include_metadata:
        fields.extend(DEFAULT_METADATA_FIELDS)
    return list(dict.fromkeys(fields))


def resolve_supported_fields(fields: list[str], valid_fields: set[str] | None) -> tuple[list[str], list[str]]:
    if valid_fields is None:
        return fields, []
    supported = [field for field in fields if field in valid_fields]
    unsupported = sorted(set(fields) - set(supported))
    if not supported:
        raise RuntimeError("None of the requested tshark fields are supported by this installation.")
    return supported, unsupported


def finalize_output_csv(output_csv: Path, fields: list[str]) -> pd.DataFrame:
    if output_csv.exists() and output_csv.stat().st_size > 0:
        df = pd.read_csv(output_csv, low_memory=False)
    else:
        df = pd.DataFrame()

    df = normalize_columns(df)
    for field in fields:
        if field not in df.columns:
            df[field] = ""
    return df[fields]


def pcap_to_csv(
    *,
    tshark: str,
    input_pcap: Path,
    output_csv: Path,
    fields: list[str],
    valid_fields: set[str] | None,
    display_filter: str | None,
    packet_count: int | None,
) -> list[str]:
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
        result = subprocess.run(command, stdout=fh, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    df = finalize_output_csv(output_csv, fields)
    df.to_csv(output_csv, index=False)
    return unsupported_fields


def prepare_model_input(df: pd.DataFrame, bundle: dict[str, object]) -> pd.DataFrame:
    df = normalize_columns(df)
    meta = bundle["training_meta"]
    feature_columns = list(meta["feature_columns"])
    numeric_columns = list(meta.get("numeric_columns", []))
    categorical_columns = list(meta.get("categorical_columns", []))

    for column in feature_columns:
        if column not in df.columns:
            df[column] = np.nan

    out = pd.DataFrame(index=df.index)
    for column in numeric_columns:
        out[column] = df[column] if column in df.columns else np.nan
    for column in categorical_columns:
        out[column] = df[column].astype("string").fillna("__MISSING__") if column in df.columns else "__MISSING__"

    typed, _, _, _ = coerce_feature_types(
        out[feature_columns],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    return typed[feature_columns]


def score_pcap_csv(
    csv_path: Path,
    bundle: dict[str, object],
    *,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame(), np.array([]), np.array([], dtype=int)

    df = pd.read_csv(csv_path, low_memory=False)
    df = normalize_columns(df)
    if df.empty:
        return df, np.array([]), np.array([], dtype=int)

    X = prepare_model_input(df, bundle)
    transformed = bundle["preprocessor"].transform(X)
    validate_runtime_contract(
        bundle=bundle,
        model_input=X,
        transformed=transformed,
        transformed_feature_names=get_transformed_feature_names(bundle["preprocessor"]),
        stage="demo_replay_score",
    )
    proba = bundle["model"].predict_proba(transformed)[:, 1]
    pred = (proba >= threshold).astype(int)
    return df, proba, pred


def build_file_summary(
    predictions: pd.DataFrame,
    *,
    file_max_threshold: float,
    file_ratio_threshold: float,
    min_records: int,
) -> pd.DataFrame:
    summary = (
        predictions.groupby("source_file", sort=True)
        .agg(
            records=("pred_label", "size"),
            malicious_records=("pred_label", "sum"),
            malicious_record_ratio=("pred_label", "mean"),
            mean_attack_probability=("pred_proba_attack", "mean"),
            median_attack_probability=("pred_proba_attack", "median"),
            p95_attack_probability=("pred_proba_attack", lambda s: float(s.quantile(0.95))),
            max_attack_probability=("pred_proba_attack", "max"),
        )
        .reset_index()
    )
    summary["file_pass_max_probability_rule"] = summary["max_attack_probability"] >= file_max_threshold
    summary["file_pass_ratio_rule"] = summary["malicious_record_ratio"] >= file_ratio_threshold
    summary["file_pred_label"] = (
        (summary["records"] >= min_records)
        & summary["file_pass_max_probability_rule"]
        & summary["file_pass_ratio_rule"]
    ).astype(int)
    return summary.sort_values(["malicious_record_ratio", "source_file"], ascending=[True, True], kind="mergesort")


def compare_summary_frames(new_summary: pd.DataFrame, reference_summary: pd.DataFrame) -> pd.DataFrame:
    new_df = new_summary.copy()
    ref_df = reference_summary.copy()

    new_df["source_key"] = new_df["source_file"].map(normalize_filename)
    ref_df["source_key"] = ref_df["source_file"].map(normalize_filename)

    merged = ref_df.merge(new_df, on="source_key", how="outer", suffixes=("_old", "_new"), indicator=True)
    merged["source_file"] = merged["source_file_old"].fillna(merged["source_file_new"])
    merged["matched_in_both"] = merged["_merge"] == "both"
    merged["file_pred_label_match"] = merged["file_pred_label_old"] == merged["file_pred_label_new"]
    merged["records_match"] = merged["records_old"] == merged["records_new"]
    merged["malicious_record_ratio_delta"] = merged["malicious_record_ratio_new"] - merged["malicious_record_ratio_old"]
    merged["mean_attack_probability_delta"] = merged["mean_attack_probability_new"] - merged["mean_attack_probability_old"]
    merged["median_attack_probability_delta"] = merged["median_attack_probability_new"] - merged["median_attack_probability_old"]
    merged["p95_attack_probability_delta"] = merged["p95_attack_probability_new"] - merged["p95_attack_probability_old"]
    merged["max_attack_probability_delta"] = merged["max_attack_probability_new"] - merged["max_attack_probability_old"]
    merged["comparison_status"] = np.select(
        [
            merged["_merge"] == "left_only",
            merged["_merge"] == "right_only",
            merged["matched_in_both"] & merged["file_pred_label_match"],
        ],
        [
            "missing_new",
            "missing_reference",
            "label_match",
        ],
        default="label_mismatch",
    )

    columns = [
        "source_file",
        "comparison_status",
        "matched_in_both",
        "file_pred_label_match",
        "records_match",
        "records_old",
        "records_new",
        "malicious_records_old",
        "malicious_records_new",
        "malicious_record_ratio_old",
        "malicious_record_ratio_new",
        "malicious_record_ratio_delta",
        "mean_attack_probability_old",
        "mean_attack_probability_new",
        "mean_attack_probability_delta",
        "median_attack_probability_old",
        "median_attack_probability_new",
        "median_attack_probability_delta",
        "p95_attack_probability_old",
        "p95_attack_probability_new",
        "p95_attack_probability_delta",
        "max_attack_probability_old",
        "max_attack_probability_new",
        "max_attack_probability_delta",
        "file_pass_max_probability_rule_old",
        "file_pass_max_probability_rule_new",
        "file_pass_ratio_rule_old",
        "file_pass_ratio_rule_new",
        "file_pred_label_old",
        "file_pred_label_new",
    ]
    return merged[columns].sort_values(["comparison_status", "source_file"], kind="mergesort")


def write_run_summary(
    *,
    output_path: Path,
    demo_dir: Path,
    reference_summary_path: Path | None,
    summary: pd.DataFrame,
    comparison: pd.DataFrame | None,
    threshold: float,
    file_max_threshold: float,
    file_ratio_threshold: float,
    unsupported_fields: list[str],
) -> None:
    lines = [
        "# Demo Replay Summary",
        "",
        f"- Demo folder: {demo_dir}",
        f"- Record threshold: {threshold:.6f}",
        f"- File max threshold: {file_max_threshold:.6f}",
        f"- File ratio threshold: {file_ratio_threshold:.6f}",
        f"- Unsupported tshark fields skipped: {len(unsupported_fields)}",
        "",
        "## Demo Files",
        f"- Files processed: {len(summary)}",
        f"- Attack labels: {int(summary['file_pred_label'].sum())}",
        f"- Benign labels: {int((summary['file_pred_label'] == 0).sum())}",
        "",
    ]

    if reference_summary_path is not None and comparison is not None:
        matched = int(comparison["matched_in_both"].sum())
        label_matches = int(comparison["file_pred_label_match"].fillna(False).sum())
        mismatches = int(((comparison["matched_in_both"]) & (~comparison["file_pred_label_match"].fillna(False))).sum())
        missing_new = int((comparison["comparison_status"] == "missing_new").sum())
        missing_ref = int((comparison["comparison_status"] == "missing_reference").sum())
        lines.extend(
            [
                "## Comparison Against Reference",
                f"- Reference summary: {reference_summary_path}",
                f"- Files matched in both: {matched}",
                f"- File-label matches: {label_matches}",
                f"- File-label mismatches: {mismatches}",
                f"- Missing in new run: {missing_new}",
                f"- Missing in reference: {missing_ref}",
                "",
                "### Matched files",
            ]
        )
        matched_files = comparison.loc[comparison["comparison_status"] == "label_match", "source_file"].tolist()
        lines.extend([f"- {name}" for name in matched_files] or ["- None"])
        lines.extend(["", "### Mismatched files"])
        mismatched_files = comparison.loc[comparison["comparison_status"] == "label_mismatch", "source_file"].tolist()
        lines.extend([f"- {name}" for name in mismatched_files] or ["- None"])
    else:
        lines.extend(
            [
                "## Comparison Against Reference",
                "- No reference summary was available, so comparison was skipped.",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_bundle(model_path: Path) -> dict[str, object]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    bundle = joblib.load(model_path)
    required = {"model", "preprocessor", "threshold", "training_meta"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {sorted(missing)}")
    attach_model_feature_names(bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline demo PCAP replay for the Edge-IIoT paper pipeline.")
    parser.add_argument("--demo_dir", default=str(DEFAULT_DEMO_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--reference_summary", default=str(DEFAULT_REFERENCE_SUMMARY))
    parser.add_argument("--tshark", default=None)
    parser.add_argument("--packet_count", type=int, default=None)
    parser.add_argument("--display_filter", default=None)
    parser.add_argument("--include_metadata", action="store_true", default=True)
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--file_max_threshold", type=float, default=None)
    parser.add_argument("--file_ratio_threshold", type=float, default=None)
    parser.add_argument("--min_records", type=int, default=0)
    args = parser.parse_args()

    try:
        started = time.perf_counter()
        demo_dir = Path(args.demo_dir)
        output_dir = Path(args.output_dir)
        extracted_dir = output_dir / "extracted_csvs"
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        tshark = find_tshark(args.tshark)
        valid_fields = available_tshark_fields(tshark)
        bundle = load_bundle(Path(args.model_path))
        threshold = float(args.threshold if args.threshold is not None else bundle["threshold"])
        file_max_threshold = float(
            args.file_max_threshold
            if args.file_max_threshold is not None
            else bundle.get("file_max_threshold", threshold)
        )
        file_ratio_threshold = float(
            args.file_ratio_threshold
            if args.file_ratio_threshold is not None
            else bundle.get("file_ratio_threshold", DEFAULT_FILE_RATIO_THRESHOLD)
        )
        include_metadata = args.include_metadata and not args.no_metadata
        requested_fields = bundle_field_contract(bundle, include_metadata)
        pcap_list = pcap_files(demo_dir)
        print(f"tshark         : {tshark}")
        print(f"Demo folder    : {demo_dir}")
        print(f"PCAP files     : {len(pcap_list)}")
        print(f"Threshold      : {threshold:.6f}")
        print(
            "File rule      : "
            f"max_probability >= {file_max_threshold:.6f} and "
            f"malicious_record_ratio >= {file_ratio_threshold:.6f}"
        )

        file_summaries: list[pd.DataFrame] = []
        per_record_frames: list[pd.DataFrame] = []
        unsupported_union: set[str] = set()
        for idx, pcap_path in enumerate(pcap_list, start=1):
            source_file = pcap_path.stem + ".csv"
            raw_csv = extracted_dir / source_file
            print(f"[{idx}/{len(pcap_list)}] {pcap_path.name} -> {raw_csv.name}")
            unsupported_fields = pcap_to_csv(
                tshark=tshark,
                input_pcap=pcap_path,
                output_csv=raw_csv,
                fields=requested_fields,
                valid_fields=valid_fields,
                display_filter=args.display_filter,
                packet_count=args.packet_count,
            )
            unsupported_union.update(unsupported_fields)

            df, proba, pred = score_pcap_csv(raw_csv, bundle, threshold=threshold)
            if df.empty:
                print(f"  skipped empty capture: {pcap_path.name}")
                continue

            per_record = pd.DataFrame(
                {
                    "source_file": source_file,
                    "record_index": np.arange(len(df)),
                    "pred_proba_attack": proba,
                    "pred_label": pred,
                }
            )
            for field in [f for f in DEFAULT_METADATA_FIELDS if f in df.columns]:
                per_record[field] = df[field].values
            per_record_frames.append(per_record)

            summary_row = build_file_summary(
                per_record,
                file_max_threshold=file_max_threshold,
                file_ratio_threshold=file_ratio_threshold,
                min_records=args.min_records,
            )
            file_summaries.append(summary_row)

        if per_record_frames:
            predictions_df = pd.concat(per_record_frames, ignore_index=True)
        else:
            predictions_df = pd.DataFrame(columns=["source_file", "record_index", "pred_proba_attack", "pred_label"])

        if file_summaries:
            summary_df = pd.concat(file_summaries, ignore_index=True)
        else:
            summary_df = pd.DataFrame(
                columns=[
                    "source_file",
                    "records",
                    "malicious_records",
                    "malicious_record_ratio",
                    "mean_attack_probability",
                    "median_attack_probability",
                    "p95_attack_probability",
                    "max_attack_probability",
                    "file_pass_max_probability_rule",
                    "file_pass_ratio_rule",
                    "file_pred_label",
                ]
            )

        predictions_path = output_dir / "edge_iiot_demo_predictions.csv"
        summary_path = output_dir / "edge_iiot_demo_predictions_summary.csv"
        comparison_path = output_dir / "edge_iiot_demo_vs_reference_comparison.csv"
        run_summary_path = output_dir / "edge_iiot_demo_run_summary.md"

        predictions_df.to_csv(predictions_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        reference_summary_path = Path(args.reference_summary) if args.reference_summary else None
        comparison_df = None
        if reference_summary_path is not None and reference_summary_path.exists():
            reference_df = pd.read_csv(reference_summary_path, low_memory=False)
            reference_df = normalize_columns(reference_df)
            comparison_df = compare_summary_frames(summary_df, reference_df)
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Comparison report: {comparison_path}")
        else:
            print(f"Reference summary not found, skipping comparison: {reference_summary_path}")

        write_run_summary(
            output_path=run_summary_path,
            demo_dir=demo_dir,
            reference_summary_path=reference_summary_path if reference_summary_path and reference_summary_path.exists() else None,
            summary=summary_df,
            comparison=comparison_df,
            threshold=threshold,
            file_max_threshold=file_max_threshold,
            file_ratio_threshold=file_ratio_threshold,
            unsupported_fields=sorted(unsupported_union),
        )

        print(f"Predictions   : {predictions_path}")
        print(f"Summary       : {summary_path}")
        print(f"Run summary   : {run_summary_path}")
        if comparison_df is not None:
            print(f"Comparison    : {comparison_path}")
        print(f"Runtime secs  : {time.perf_counter() - started:.3f}")

        if not summary_df.empty:
            print("\nFile summary:")
            print(summary_df.to_string(index=False))
        if comparison_df is not None and not comparison_df.empty:
            print("\nComparison summary:")
            print(comparison_df.to_string(index=False))

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
