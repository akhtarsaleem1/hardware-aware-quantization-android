#!/usr/bin/env python3
"""Select validation/device Pareto fronts before the test split is opened."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


VARIANTS = ("float32", "float16", "dynamic_int8", "full_int8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variant_name(value: str) -> str:
    name = Path(value).stem.lower()
    matches = [variant for variant in VARIANTS if variant in name]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify variant from {value!r}")
    return matches[0]


def parse_parity(value: str) -> tuple[str, Path]:
    architecture, separator, path = value.partition("=")
    if not separator or not architecture or not path:
        raise argparse.ArgumentTypeError("use ARCHITECTURE=PARITY_REPORT.json")
    return architecture, Path(path)


def pareto(records: list[dict[str, object]], latency: str) -> list[str]:
    values = np.asarray(
        [
            [
                float(record["validation_accuracy"]),
                -float(record["artifact_size_bytes"]),
                -float(record[latency]),
            ]
            for record in records
        ],
        dtype=float,
    )
    selected: list[str] = []
    for index, record in enumerate(records):
        at_least = np.all(values >= values[index], axis=1)
        strict = np.any(values > values[index], axis=1)
        if not np.any(at_least & strict):
            selected.append(str(record["configuration_id"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--benchmark-audit", required=True, type=Path)
    parser.add_argument("--app-manifest", required=True, type=Path)
    parser.add_argument("--flutter-root", type=Path, default=Path("flutter_app"))
    parser.add_argument("--parity", required=True, action="append", type=parse_parity)
    parser.add_argument("--expected-runs", type=int, default=100)
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("results/pretest/selection_summary.json")
    )
    args = parser.parse_args()
    if len(args.parity) != 3 or len({item[0] for item in args.parity}) != 3:
        raise SystemExit("Exactly three distinct parity reports are required")

    benchmark_audit = json.loads(args.benchmark_audit.read_text(encoding="utf-8"))
    if benchmark_audit.get("status") != "PASS":
        raise ValueError("Independent benchmark audit is not PASS")
    if benchmark_audit.get("raw_csv_sha256") != sha256(args.raw):
        raise ValueError("Benchmark audit does not hash-link the selected raw CSV")
    if benchmark_audit.get("test_split_accessed") is not False:
        raise ValueError("Benchmark audit does not attest that the test split stayed locked")
    if (
        int(benchmark_audit.get("successful_configurations", -1))
        + int(benchmark_audit.get("failed_configurations", -1))
        != 72
    ):
        raise ValueError("Benchmark audit does not account for all 72 configurations")

    validation: dict[tuple[str, str], dict[str, float]] = {}
    parity_provenance: list[dict[str, str]] = []
    for architecture, path in args.parity:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") not in (
            "VALIDATION_ONLY_TEST_SPLIT_UNTOUCHED",
            "VALIDATION_PARITY_ONLY_TEST_SPLIT_UNTOUCHED",
        ):
            raise ValueError(f"Parity report is not validation-only: {path}")
        for key, metrics in report["variants"].items():
            quantization = variant_name(str(metrics.get("path", key)))
            accuracy = metrics.get("validation_accuracy", metrics.get("accuracy"))
            agreement = metrics.get("top1_agreement_with_keras")
            if accuracy is None or agreement is None:
                raise ValueError(f"Incomplete validation metrics: {architecture}/{key}")
            validation[(architecture, quantization)] = {
                "accuracy": float(accuracy),
                "agreement": float(agreement),
            }
        parity_provenance.append(
            {"architecture": architecture, "path": str(path), "sha256": sha256(path)}
        )
    if len(validation) != 12:
        raise ValueError(f"Expected 12 validation variants, found {len(validation)}")

    app = json.loads(args.app_manifest.read_text(encoding="utf-8"))
    if app.get("status") != "FROZEN_FINAL_DEVICE_BENCHMARK":
        raise ValueError("App manifest is not frozen for the final device benchmark")
    if app.get("protocol_version") != "1.2.0":
        raise ValueError("Unexpected app protocol version")
    app_models = {model["id"]: model for model in app["models"]}
    if len(app_models) != 12:
        raise ValueError("Expected 12 distinct app model configurations")
    for model in app_models.values():
        asset = args.flutter_root / model["asset"]
        if sha256(asset) != model["sha256"]:
            raise ValueError(f"App asset hash mismatch: {asset}")

    measured: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    errors: dict[tuple[str, str, int, int], list[str]] = defaultdict(list)
    with args.raw.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if row.get("protocol_version") != "1.2.0":
                raise ValueError(f"Raw row {line_number} has wrong protocol version")
            if row.get("app_version") != "1.2.0+3" or row.get("build_mode") != "release":
                raise ValueError(f"Raw row {line_number} is not from the final release app")
            if row.get("battery_saver", "").lower() != "false":
                raise ValueError(f"Battery saver was not off on raw row {line_number}")
            expected_delegate = (
                "xnnpack_initialized_partition_unverified"
                if row.get("runtime") == "xnnpack_cpu"
                else "none_builtin"
            )
            if row.get("effective_delegate") != expected_delegate:
                raise ValueError(f"Unexpected delegate label on raw row {line_number}")
            model_id = row.get("model_id", "")
            if model_id not in app_models:
                raise ValueError(f"Unknown model id on raw row {line_number}: {model_id}")
            key = (
                model_id,
                row.get("runtime", ""),
                int(row["threads"]),
                int(row["trial_id"]),
            )
            if row.get("model_sha256") != app_models[model_id]["sha256"]:
                raise ValueError(f"Raw model hash mismatch on line {line_number}")
            if row.get("phase") == "measured":
                latency = float(row["latency_ms"])
                if not np.isfinite(latency) or latency <= 0:
                    raise ValueError(f"Invalid latency on line {line_number}")
                measured[key].append(latency)
            elif row.get("phase") == "configuration_error":
                errors[key].append(row.get("error", ""))
            else:
                raise ValueError(f"Unsupported phase on line {line_number}")

    records: list[dict[str, object]] = []
    preserved_errors: list[dict[str, object]] = []
    expected_keys = {
        (model_id, runtime, threads, trial)
        for model_id in app_models
        for runtime in app["runtimes"]
        for threads in app["threads"]
        for trial in range(1, args.expected_trials + 1)
    }
    if set(measured) | set(errors) != expected_keys:
        missing = sorted(expected_keys - (set(measured) | set(errors)))
        extra = sorted((set(measured) | set(errors)) - expected_keys)
        raise ValueError(f"Raw configuration coverage mismatch; missing={missing}, extra={extra}")

    for model_id, model in sorted(app_models.items()):
        for runtime in app["runtimes"]:
            for threads in app["threads"]:
                keys = [
                    (model_id, runtime, threads, trial)
                    for trial in range(1, args.expected_trials + 1)
                ]
                if all(len(measured[key]) == args.expected_runs and not errors[key] for key in keys):
                    trial_medians = np.asarray(
                        [np.median(measured[key]) for key in keys], dtype=float
                    )
                    trial_p95 = np.asarray(
                        [np.percentile(measured[key], 95) for key in keys], dtype=float
                    )
                    evidence = validation[(model["architecture"], model["quantization"])]
                    asset = args.flutter_root / model["asset"]
                    records.append(
                        {
                            "configuration_id": f"{model_id}__{runtime}__{threads}t",
                            "model_id": model_id,
                            "architecture": model["architecture"],
                            "quantization": model["quantization"],
                            "runtime": runtime,
                            "threads": threads,
                            "validation_accuracy": evidence["accuracy"],
                            "validation_top1_agreement": evidence["agreement"],
                            "artifact_size_bytes": asset.stat().st_size,
                            "trial_median_mean_ms": float(trial_medians.mean()),
                            "trial_median_sd_ms": float(trial_medians.std(ddof=1)),
                            "trial_p95_mean_ms": float(trial_p95.mean()),
                            "trial_p95_sd_ms": float(trial_p95.std(ddof=1)),
                        }
                    )
                elif all(not measured[key] and len(errors[key]) == 1 for key in keys):
                    preserved_errors.append(
                        {
                            "configuration_id": f"{model_id}__{runtime}__{threads}t",
                            "trial_errors": [errors[key][0] for key in keys],
                        }
                    )
                else:
                    raise ValueError(
                        f"Mixed/incomplete success state: {model_id}/{runtime}/{threads}t"
                    )

    output = {
        "status": "PRETEST_SELECTION_COMPLETE",
        "test_split_accessed": False,
        "selection_quality_source": "validation only",
        "raw_benchmark": str(args.raw),
        "raw_benchmark_sha256": sha256(args.raw),
        "benchmark_audit": str(args.benchmark_audit),
        "benchmark_audit_sha256": sha256(args.benchmark_audit),
        "app_manifest": str(args.app_manifest),
        "app_manifest_sha256": sha256(args.app_manifest),
        "parity_reports": parity_provenance,
        "successful_configuration_count": len(records),
        "failed_configuration_count": len(preserved_errors),
        "median_pareto_configuration_ids": pareto(records, "trial_median_mean_ms"),
        "p95_pareto_configuration_ids": pareto(records, "trial_p95_mean_ms"),
        "configuration_records": records,
        "configuration_errors": preserved_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
