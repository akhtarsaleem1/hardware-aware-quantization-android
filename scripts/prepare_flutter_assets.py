#!/usr/bin/env python3
"""Bundle verified flatbuffers and build the frozen Flutter model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


VARIANTS = ("float32", "float16", "dynamic_int8", "full_int8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_model(value: str) -> tuple[str, Path]:
    architecture, separator, report = value.partition("=")
    if not separator or not architecture or not report:
        raise argparse.ArgumentTypeError("use ARCHITECTURE=VERIFICATION_REPORT.json")
    return architecture, Path(report)


def identify_variant(key: str) -> str:
    matches = [variant for variant in VARIANTS if key.endswith(f"__{variant}")]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify exactly one variant in {key!r}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=parse_model,
        help="ARCHITECTURE=flatbuffer_verification.json; repeat for each model",
    )
    parser.add_argument("--flutter-root", type=Path, default=Path("flutter_app"))
    parser.add_argument(
        "--bundle-report",
        type=Path,
        default=Path("reports/flutter_model_bundle.json"),
    )
    args = parser.parse_args()

    if len(args.model) != 3 or len({name for name, _ in args.model}) != 3:
        raise SystemExit("Exactly three distinct architecture reports are required")
    assets = args.flutter_root / "assets" / "models"
    assets.mkdir(parents=True, exist_ok=True)
    models: list[dict[str, object]] = []
    report_provenance: list[dict[str, str]] = []

    for architecture, report_path in sorted(args.model):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise ValueError(f"Verification report is not PASS: {report_path}")
        found: dict[str, dict[str, object]] = {}
        for key, value in report["variants"].items():
            variant = identify_variant(key)
            if variant in found:
                raise ValueError(f"Duplicate {architecture}/{variant}")
            if value.get("status") != "success":
                raise ValueError(f"Unsuccessful {architecture}/{variant}")
            found[variant] = value
        if set(found) != set(VARIANTS):
            raise ValueError(f"Incomplete variants for {architecture}: {sorted(found)}")

        report_provenance.append(
            {"architecture": architecture, "path": str(report_path), "sha256": sha256(report_path)}
        )
        for variant in VARIANTS:
            value = found[variant]
            source = Path(str(value["path"]))
            if not source.is_file():
                raise FileNotFoundError(source)
            if source.stat().st_size != int(value["size_bytes"]):
                raise ValueError(f"Size mismatch: {source}")
            source_hash = sha256(source)
            if source_hash != value["sha256"]:
                raise ValueError(f"Hash mismatch: {source}")
            inputs = value["inputs"]
            outputs = value["outputs"]
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError(f"Expected one input and output: {source}")
            if inputs[0]["shape"] != [1, 224, 224, 3] or outputs[0]["shape"] != [1, 9]:
                raise ValueError(f"Unexpected tensor shape: {source}")

            filename = f"{architecture}_{variant}.tflite"
            destination = assets / filename
            if not destination.is_file() or sha256(destination) != source_hash:
                shutil.copy2(source, destination)
            if sha256(destination) != source_hash:
                raise RuntimeError(f"Copied asset failed hash verification: {destination}")
            models.append(
                {
                    "id": f"{architecture}_{variant}",
                    "asset": f"assets/models/{filename}",
                    "sha256": source_hash,
                    "architecture": architecture,
                    "quantization": variant,
                    "input_dtype": inputs[0]["dtype"],
                    "output_dtype": outputs[0]["dtype"],
                }
            )

    manifest = {
        "protocol_version": "1.2.0",
        "status": "FROZEN_FINAL_DEVICE_BENCHMARK",
        "warmup_runs": 20,
        "measured_runs": 100,
        "complete_trials": 3,
        "random_seed": 42,
        "threads": [1, 2, 4],
        "runtimes": ["builtin_cpu", "xnnpack_cpu"],
        "models": models,
    }
    manifest_path = args.flutter_root / "assets" / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bundle = {
        "status": "PASS",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "model_count": len(models),
        "asset_count": len(list(assets.glob("*.tflite"))),
        "verification_reports": report_provenance,
        "models": models,
    }
    if bundle["asset_count"] != 12:
        raise ValueError(
            f"Expected exactly 12 TFLite assets, found {bundle['asset_count']}"
        )
    args.bundle_report.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_report.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    main()
