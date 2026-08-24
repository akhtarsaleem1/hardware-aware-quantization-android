#!/usr/bin/env python3
"""Freeze model and validation evidence before the single locked-test session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


VARIANTS = ("float32", "float16", "dynamic_int8", "full_int8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_spec(value: str) -> tuple[str, Path, Path, Path, Path]:
    architecture, separator, paths = value.partition("=")
    values = paths.split(",")
    if not separator or not architecture or len(values) != 4:
        raise argparse.ArgumentTypeError(
            "use ARCH=KERAS,TFLITE_DIR,PARITY_REPORT,VERIFICATION_REPORT"
        )
    return architecture, *(Path(item) for item in values)


def variant_name(value: str) -> str:
    name = Path(value).stem.lower()
    matches = [variant for variant in VARIANTS if name.endswith(f"__{variant}")]
    if len(matches) != 1:
        matches = [variant for variant in VARIANTS if variant in name]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify one variant from {value!r}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, type=parse_spec)
    parser.add_argument(
        "--split-metadata",
        type=Path,
        default=Path("data/manifests_grouped/manifest_metadata.json"),
    )
    parser.add_argument("--pretest-selection", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/final_test_manifest.json")
    )
    args = parser.parse_args()
    if len(args.model) != 3 or len({value[0] for value in args.model}) != 3:
        raise SystemExit("Exactly three distinct architecture specifications are required")

    split_metadata = json.loads(args.split_metadata.read_text(encoding="utf-8"))
    if split_metadata.get("status") != "FROZEN_PRETRAINING_SPLIT":
        raise ValueError("Dataset split metadata is not frozen")
    pretest = json.loads(args.pretest_selection.read_text(encoding="utf-8"))
    if pretest.get("status") != "PRETEST_SELECTION_COMPLETE":
        raise ValueError("Pre-test selection report is not complete")
    if pretest.get("test_split_accessed") is not False:
        raise ValueError("Pre-test selection does not attest locked test data")
    if not pretest.get("median_pareto_configuration_ids"):
        raise ValueError("Pre-test median Pareto front is empty")
    if not pretest.get("p95_pareto_configuration_ids"):
        raise ValueError("Pre-test p95 Pareto front is empty")
    models: list[dict[str, object]] = []
    for architecture, keras_path, tflite_dir, parity_path, verification_path in sorted(
        args.model
    ):
        if not keras_path.is_file() or not tflite_dir.is_dir():
            raise FileNotFoundError(f"Missing model artifact for {architecture}")
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if parity.get("status") not in (
            "VALIDATION_ONLY_TEST_SPLIT_UNTOUCHED",
            "VALIDATION_PARITY_ONLY_TEST_SPLIT_UNTOUCHED",
        ):
            raise ValueError(f"Parity report not validation-only: {parity_path}")
        if verification.get("status") != "PASS":
            raise ValueError(f"Flatbuffer verification not PASS: {verification_path}")

        verified: dict[str, dict[str, object]] = {}
        for key, metrics in verification["variants"].items():
            variant = variant_name(key)
            if metrics.get("status") != "success":
                raise ValueError(f"Failed verified variant {architecture}/{variant}")
            verified[variant] = metrics
        parity_variants: dict[str, dict[str, object]] = {}
        for key, metrics in parity["variants"].items():
            variant = variant_name(str(metrics.get("path", key)))
            parity_variants[variant] = metrics
        if set(verified) != set(VARIANTS) or set(parity_variants) != set(VARIANTS):
            raise ValueError(f"Incomplete variants for {architecture}")

        validation_variants: dict[str, dict[str, object]] = {}
        for variant in VARIANTS:
            verification_metrics = verified[variant]
            parity_metrics = parity_variants[variant]
            flatbuffer = Path(str(verification_metrics["path"]))
            if flatbuffer.parent.resolve() != tflite_dir.resolve():
                raise ValueError(f"Verified path outside frozen directory: {flatbuffer}")
            actual_hash = sha256(flatbuffer)
            if actual_hash != verification_metrics["sha256"]:
                raise ValueError(f"Flatbuffer hash mismatch: {flatbuffer}")
            accuracy = parity_metrics.get(
                "validation_accuracy", parity_metrics.get("accuracy")
            )
            agreement = parity_metrics.get("top1_agreement_with_keras")
            if accuracy is None or agreement is None:
                raise ValueError(f"Incomplete parity metrics: {architecture}/{variant}")
            validation_variants[variant] = {
                "path": str(flatbuffer),
                "sha256": actual_hash,
                "size_bytes": flatbuffer.stat().st_size,
                "validation_accuracy": float(accuracy),
                "top1_agreement_with_keras": float(agreement),
            }
        models.append(
            {
                "architecture": architecture,
                "keras_model": str(keras_path),
                "keras_sha256": sha256(keras_path),
                "tflite_dir": str(tflite_dir),
                "parity_report": str(parity_path),
                "parity_report_sha256": sha256(parity_path),
                "verification_report": str(verification_path),
                "verification_report_sha256": sha256(verification_path),
                "validation_variants": validation_variants,
            }
        )

    manifest = {
        "status": "FROZEN_FOR_FINAL_TEST",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluation_count_before_freeze": 0,
        "selection_uses": [
            "training evidence",
            "validation accuracy and top-1 agreement",
            "flatbuffer allocation/execution",
            "frozen Android configuration protocol",
            "completed validation/device pre-test Pareto analysis",
        ],
        "selection_prohibits": ["test accuracy", "test macro F1", "test predictions"],
        "dataset_split_protocol": split_metadata["method"],
        "test_manifest_record_count": split_metadata["split_counts"]["test"],
        "test_manifest_sha256_from_pretraining_metadata": split_metadata[
            "manifest_sha256"
        ]["test"],
        "calibration": {
            "source": "training split only",
            "sample_count": 800,
            "npz": "data/calibration/deepweeds_train_stratified_seed42_n800.npz",
            "npz_sha256": sha256(
                Path("data/calibration/deepweeds_train_stratified_seed42_n800.npz")
            ),
        },
        "benchmark_protocol": "benchmark/benchmark_config.yaml",
        "benchmark_protocol_sha256": sha256(Path("benchmark/benchmark_config.yaml")),
        "pretest_selection_report": str(args.pretest_selection),
        "pretest_selection_report_sha256": sha256(args.pretest_selection),
        "raw_benchmark": pretest["raw_benchmark"],
        "raw_benchmark_sha256": pretest["raw_benchmark_sha256"],
        "benchmark_audit": pretest["benchmark_audit"],
        "benchmark_audit_sha256": pretest["benchmark_audit_sha256"],
        "median_pareto_configuration_ids": pretest[
            "median_pareto_configuration_ids"
        ],
        "p95_pareto_configuration_ids": pretest["p95_pareto_configuration_ids"],
        "preserved_device_configuration_error_count": pretest[
            "failed_configuration_count"
        ],
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
