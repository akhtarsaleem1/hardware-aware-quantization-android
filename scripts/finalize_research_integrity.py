#!/usr/bin/env python3
"""Build the final integrity report and result-level provenance ledger."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


FIELDS = [
    "result_id", "source_file", "experiment_id", "raw_data_file",
    "processing_script", "date", "device", "model", "quantization",
    "runtime", "status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def require(path: Path, status: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != status:
        raise ValueError(f"{path} status is not {status}")
    return value


def row(
    result_id: str, source: str, experiment: str, raw: str, script: str,
    date: str, device: str, model: str = "", quantization: str = "",
    runtime: str = "", status: str = "reported",
) -> dict[str, str]:
    return dict(zip(FIELDS, [
        result_id, source, experiment, raw, script, date, device, model,
        quantization, runtime, status,
    ]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    audit_path = root / "reports/final_benchmark_audit.json"
    analysis_path = root / "results/final/analysis_summary.json"
    test_path = root / "results/final/final_test_report.json"
    supplement_path = root / "results/final/publication_supplement/supplement_manifest.json"
    audit = require(audit_path, "PASS")
    analysis = require(analysis_path, "FINAL_ANALYSIS_COMPLETE")
    test = require(test_path, "FINAL_TEST_SINGLE_SESSION_COMPLETE")
    supplement = require(supplement_path, "PUBLICATION_SUPPLEMENT_COMPLETE")
    if analysis.get("raw_csv_sha256") != audit.get("raw_csv_sha256"):
        raise ValueError("analysis and benchmark audit raw hashes differ")
    raw_path = resolve(root, str(analysis["raw_csv"]))
    if sha256(raw_path) != audit["raw_csv_sha256"]:
        raise ValueError("final raw CSV hash mismatch")
    if test.get("sample_count") != 3501:
        raise ValueError("locked-test report does not contain 3,501 samples")

    configurations = pd.read_csv(root / "results/final/table_configuration_results.csv")
    model_results = pd.read_csv(root / "results/final/table_model_results.csv")
    errors = pd.read_csv(root / "results/final/table_configuration_errors.csv")
    if len(model_results) != 12:
        raise ValueError("model result table does not contain 12 variants")
    if len(configurations) != int(audit["successful_configurations"]):
        raise ValueError("configuration table count differs from benchmark audit")

    date = str(audit["device_finished_utc"])[:10]
    rows: list[dict[str, str]] = [
        row(
            "dataset_grouped_split", "reports/dataset_report.json",
            "grouped_capture_session_v1_gap90_seed42",
            "data/manifests_grouped/train.csv;data/manifests_grouped/validation.csv;data/manifests_grouped/test.csv",
            "scripts/create_grouped_split.py;scripts/validate_dataset.py",
            "2026-08-21", "workstation",
        ),
        row(
            "android_raw_audit", "reports/final_benchmark_audit.json",
            "android_protocol_1_2_bounded_trials_1_2_exploratory", str(raw_path.relative_to(root)),
            "scripts/audit_final_benchmark.py", date, str(audit["device_id"]),
        ),
        row(
            "locked_test_session", "results/final/final_test_report.json",
            "single_session_locked_test", "results/final/final_test_predictions.csv",
            "scripts/evaluate_final_test.py", date,
            "NVIDIA_GTX_1650_Keras_and_builtin_CPU_TFLite",
        ),
    ]
    for record in model_results.itertuples():
        rows.append(row(
            f"model_{record.architecture}_{record.quantization}",
            "results/final/table_model_results.csv",
            "locked_test_and_frozen_validation",
            "results/final/final_test_predictions.csv",
            "scripts/evaluate_final_test.py;scripts/analyze_final_results.py",
            date, "NVIDIA_GTX_1650_Keras_and_builtin_CPU_TFLite",
            str(record.architecture), str(record.quantization),
        ))
    for record in configurations.itertuples():
        rows.append(row(
            f"config_{record.configuration_id}",
            "results/final/table_configuration_results.csv",
            "android_protocol_1_2_bounded_trials_1_2_exploratory", str(raw_path.relative_to(root)),
            "scripts/audit_final_benchmark.py;scripts/analyze_final_results.py",
            date, str(audit["device_id"]), str(record.architecture),
            str(record.quantization), f"{record.runtime}/{record.threads}t",
        ))
    for record in errors.itertuples():
        rows.append(row(
            f"error_{record.model_id}_{record.runtime}_{record.threads}t_trial{record.trial_id}",
            "results/final/table_configuration_errors.csv",
            "android_protocol_1_2_bounded_trials_1_2_exploratory", str(raw_path.relative_to(root)),
            "scripts/audit_final_benchmark.py;scripts/analyze_final_results.py",
            date, str(audit["device_id"]), str(record.architecture),
            str(record.quantization), f"{record.runtime}/{record.threads}t",
            "preserved_configuration_error",
        ))
    for artifact in supplement["artifacts"]:
        path = str(artifact["path"])
        rows.append(row(
            "supplement_" + Path(path).name.replace(".", "_"), path,
            "publication_derivation",
            str(raw_path.relative_to(root)) + ";results/final/final_test_report.json",
            "scripts/generate_publication_supplement.py", date,
            str(audit["device_id"]),
        ))
    figure_names = {
        1: "accuracy_size", 2: "latency", 3: "latency",
        4: "pareto", 5: "rank_stability",
    }
    for number, name in figure_names.items():
        path = f"results/final/figures/figure_{number}_{name}.png"
        rows.append(row(
            f"manuscript_figure_{number}", path, "final_analysis",
            str(raw_path.relative_to(root)) + ";results/final/final_test_report.json",
            "scripts/analyze_final_results.py", date, str(audit["device_id"]),
        ))

    provenance_path = root / "research_integrity/result_provenance.csv"
    with provenance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    failed_11 = require(
        root / "benchmark/failed_runs/protocol_1_1_anr_20260822/failure_report.json",
        "FAILED_INCOMPLETE_EXCLUDED_FROM_ANALYSIS",
    )
    failed_12 = require(
        root / "benchmark/failed_runs/protocol_1_2_watermark_pressure_20260822/failure_report.json",
        "FAILED_INCOMPLETE_EXCLUDED_FROM_ANALYSIS",
    )
    failed_13 = require(
        root / "benchmark/failed_runs/protocol_1_2_foreground_loss_20260822/failure_report.json",
        "FAILED_INCOMPLETE_EXCLUDED_FROM_ANALYSIS",
    )
    failed_final = require(
        root / "benchmark/failed_runs/protocol_1_2_final_candidate_incomplete_20260822/failure_report.json",
        "FAILED_INCOMPLETE_EXCLUDED_FROM_CONFIRMATORY_ANALYSIS",
    )
    report = f"""# Research integrity report

Updated: {datetime.now(timezone.utc).date().isoformat()}

## Actually completed

- Verified 34 real literature records and 34 unique bibliography entries; all
  manuscript citation keys resolve.
- Validated all 17,509 DeepWeeds images and froze a capture-session-grouped
  10,506/3,502/3,501 split with no unresolved cross-split candidate.
- Trained MobileNetV2, MobileNetV3-Small, and EfficientNet-B0 under a
  fail-closed TensorFlow 2.21 WSL2 policy on the NVIDIA GeForce GTX 1650.
- Exported and hash-verified FP32, FP16, dynamic INT8, and full INT8
  flatbuffers using one 800-sample training-only calibration artifact.
- Built and tested a local-only Flutter research prototype and release
  benchmark APK (protocol 1.2.0, app 1.2.0+3).
- Completed and independently audited a post hoc bounded Android dataset with
  two complete balanced trials, {int(audit['measured_rows']):,} measured rows,
  {int(audit['successful_configurations'])} successful configurations, and
  {int(audit['configuration_error_rows'])} explicit error rows on
  {audit['device_id']}.
- Froze validation/device Pareto selection before opening the test set, then
  performed one locked 3,501-image final-test session with Keras references
  verified on /GPU:0.
- Generated hash-linked tables, high-resolution/vector figures, a populated
  manuscript, supporting statements, and {len(rows)} provenance rows.

## Failed experiments and negative evidence

- EfficientNet-B0 batch 16 produced a recorded GPU ResourceExhaustedError;
  batch 8 was frozen only after all architectures passed GPU feasibility.
- MobileNetV3-Small full INT8 had poor validation parity and was retained.
- Protocol 1.1 ended in an ANR after synchronous LiteRT work occupied the UI
  isolate. Its {int(failed_11['data_rows_preserved']):,}-row partial CSV is hashed and
  excluded.
- A protocol-1.2 attempt was paused when another app took foreground; Android
  killed PID {failed_12['android_event_evidence']['benchmark_pid']} for
  Watermark Pressure. Its {int(failed_12['data_rows']):,}-row partial CSV is
  hashed and excluded. It was never resumed, pooled, or imputed.
- A second protocol-1.2 candidate lost the foreground condition to
  {failed_13['unexpected_top_activity']}. Its {int(failed_13['data_rows']):,}-row
  partial CSV is hashed and excluded; the process was stopped without reuse.
- The final protocol-1.2 candidate stopped during trial 3. Its
  {int(failed_final['source_data_rows']):,}-row source CSV is hashed and retained;
  trial 3 was excluded wholesale, with no pooling or imputation. Only balanced
  trials 1-2 enter the explicitly exploratory bounded analysis.
- Final configuration failures remain explicit; no fallback measurements were
  introduced.

## Not completed or not claimed

- NNAPI was excluded because effective execution could not be verified on this
  Android 15 stack and the available API path was deprecated.
- Energy use was not attributed per model; no cold-start claim is made.
- No second physical phone was available, so cross-device generalization and
  hardware-population significance are not claimed.
- No direct live Clarivate profile was captured; SCIE indexing is not claimed.
- Author metadata, declarations, sole-author approval, and a reserved Zenodo
  DOI were supplied on 2026-08-24. The Zenodo record still requires publication
  before journal submission.

## Assumptions, bias risks, and limitations

- One phone, two CPU paths, one dataset, three architectures, and post-training
  quantization limit generalization.
- USB charging and uncontrolled background apps are recorded confounders.
- PSS/RSS are observational process snapshots, not isolated peak allocations.
- Two complete balanced trials support only descriptive repeatability and
  imprecise device-level uncertainty; the planned third trial was excluded
  wholesale after the run became incomplete. This post hoc bounded analysis is
  exploratory, not confirmatory, and invocations are not independent device
  replicates.
- Pareto quality uses frozen validation accuracy; locked-test metrics were not
  used to choose configurations.

## Reproducibility status

Scientific-package reproducibility is complete for the recorded workstation and
phone stack. The audited raw CSV SHA-256 is {audit['raw_csv_sha256']}; the
locked prediction CSV SHA-256 is {test['prediction_file_sha256']}. Every
supplement artifact is hashed in its supplement manifest.
"""
    (root / "research_integrity/report.md").write_text(report, encoding="utf-8")

    selection_path = root / "reports/final_test_manifest.json"
    selection = require(selection_path, "FROZEN_FOR_FINAL_TEST")
    app_manifest_path = root / "flutter_app/assets/benchmark_manifest.json"
    apk_report = require(root / "reports/final_apk.json", "PASS")
    gpu_preflight_path = root / "environment/final_gpu_preflight_20260822.json"
    gpu_preflight = require(gpu_preflight_path, "PASS")
    dataset_report_path = root / "reports/dataset_report.json"
    dataset_report = require(dataset_report_path, "PASS")
    experiment_manifest = {
        "status": "EXPERIMENT_COMPLETE_SCIENTIFIC_PACKAGE_READY",
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
        "dataset": {
            "name": "DeepWeeds",
            "archive_sha256": "0961f63c01b997bfab1559ad09e99c0e8130617fd96a8b92fdc09940e01b0ce8",
            "split_protocol": "grouped_capture_session_v1_gap90_seed42",
            "report": str(dataset_report_path.relative_to(root)),
            "report_sha256": sha256(dataset_report_path),
            "split_counts": dataset_report["split_counts"],
            "manifest_hashes": {
                split: sha256(root / f"data/manifests_grouped/{split}.csv")
                for split in ("train", "validation", "test")
            },
        },
        "gpu": {
            "policy": "GPU_REQUIRED_ABORT_ON_CPU_FALLBACK",
            "final_preflight": str(gpu_preflight_path.relative_to(root)),
            "final_preflight_sha256": sha256(gpu_preflight_path),
            "tensorflow": gpu_preflight["tensorflow"],
            "device": gpu_preflight["nvidia_smi"],
        },
        "calibration": selection["calibration"],
        "models": selection["models"],
        "android": {
            "device_id": audit["device_id"],
            "protocol_version": "1.2.0",
            "app_version": "1.2.0+3",
            "app_manifest": str(app_manifest_path.relative_to(root)),
            "app_manifest_sha256": sha256(app_manifest_path),
            "release_apk_sha256": apk_report["sha256"],
            "raw_csv": str(raw_path.relative_to(root)),
            "raw_csv_sha256": audit["raw_csv_sha256"],
            "benchmark_audit": str(audit_path.relative_to(root)),
            "benchmark_audit_sha256": sha256(audit_path),
            "successful_configurations": audit["successful_configurations"],
            "failed_configurations": audit["failed_configurations"],
            "measured_rows": audit["measured_rows"],
        },
        "final_test": {
            "evaluation_count": 1,
            "manifest": str(selection_path.relative_to(root)),
            "manifest_sha256": sha256(selection_path),
            "report": str(test_path.relative_to(root)),
            "report_sha256": sha256(test_path),
            "predictions": test["prediction_file"],
            "predictions_sha256": test["prediction_file_sha256"],
        },
        "analysis": {
            "summary": str(analysis_path.relative_to(root)),
            "summary_sha256": sha256(analysis_path),
            "pareto_selection_uses_validation_not_test": True,
        },
        "publication_supplement": {
            "manifest": str(supplement_path.relative_to(root)),
            "manifest_sha256": sha256(supplement_path),
            "table_count": supplement["table_count"],
            "figure_pdf_count": supplement["figure_pdf_count"],
        },
        "excluded_failed_runs": [
            {
                "path": path,
                "sha256": sha256(root / path),
                "status": "FAILED_INCOMPLETE_EXCLUDED_FROM_ANALYSIS",
            }
            for path in (
                "benchmark/failed_runs/protocol_1_1_anr_20260822/failure_report.json",
                "benchmark/failed_runs/protocol_1_2_watermark_pressure_20260822/failure_report.json",
                "benchmark/failed_runs/protocol_1_2_foreground_loss_20260822/failure_report.json",
            )
        ],
        "provenance": {
            "path": str(provenance_path.relative_to(root)),
            "sha256": sha256(provenance_path),
            "rows": len(rows),
        },
        "novelty_position": "FINAL_RECHECK_COMPLETE_REPLICATION_EXTENSION_FRAMING",
        "clarivate_scie_directly_verified": False,
        "git_commit": "NOT_AVAILABLE_WORKSPACE_ROOT_NOT_A_GIT_REPOSITORY",
    }
    experiment_manifest_path = root / "reports/experiment_manifest.json"
    experiment_manifest_path.write_text(
        json.dumps(experiment_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "RESEARCH_INTEGRITY_FINALIZED",
        "provenance_rows": len(rows),
        "raw_benchmark_sha256": audit["raw_csv_sha256"],
        "provenance_sha256": sha256(provenance_path),
        "experiment_manifest_sha256": sha256(experiment_manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
