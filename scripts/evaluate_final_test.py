#!/usr/bin/env python3
"""Single-session final evaluation on the locked test split.

Do not run this script until every model/quantization choice is frozen. It emits
both aggregate JSON metrics and immutable per-sample predictions. A start record
is created with exclusive semantics before test rows are read; any subsequent
invocation refuses to run, including after an interrupted first invocation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deepweeds import CLASS_NAMES
from src.data.tf_pipeline import build_dataset, read_manifest
from src.quantization.tensor_io import dequantize_output, quantize_input
from src.training.device import assert_tensor_on_gpu, require_gpu


VARIANTS = ("float32", "float16", "dynamic_int8", "full_int8")
EXPECTED_ARCHITECTURES = {
    "efficientnet_b0",
    "mobilenet_v2",
    "mobilenet_v3_small",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_root(raw_value: str) -> Path:
    prefix = "${DATASET_PATH:-"
    if raw_value.startswith(prefix) and raw_value.endswith("}"):
        return Path(os.environ.get("DATASET_PATH", raw_value[len(prefix) : -1]))
    return Path(os.path.expandvars(raw_value))


def _metrics(expected: list[int], predicted: list[int]) -> dict[str, object]:
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(
        expected, predicted, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        "per_class": {
            str(index): {
                "name": CLASS_NAMES[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }


def _frozen_path(value: object) -> Path:
    """Interpret frozen project-relative paths on Windows and POSIX/WSL."""
    return Path(str(value).replace("\\", "/"))


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"frozen {label} hash mismatch: {path}")


def _variant_name(value: str) -> str:
    name = Path(value).stem.lower()
    matches = [variant for variant in VARIANTS if name.endswith(f"__{variant}")]
    if len(matches) != 1:
        raise ValueError(f"cannot identify frozen TFLite variant from {value!r}")
    return matches[0]


def _preflight(
    experiment: dict[str, object],
    config: dict[str, object],
    manifest_path: Path,
) -> list[dict[str, object]]:
    if experiment.get("status") != "FROZEN_FOR_FINAL_TEST":
        raise SystemExit("manifest status must be FROZEN_FOR_FINAL_TEST")
    if experiment.get("test_evaluation_count_before_freeze") != 0:
        raise ValueError("frozen manifest does not attest zero prior test evaluations")
    if not config["dataset"].get("split_protocol") or not experiment.get(
        "dataset_split_protocol"
    ):
        raise ValueError("configured or frozen split protocol identifier is missing")
    _require_hash(
        manifest_path,
        str(experiment["test_manifest_sha256_from_pretraining_metadata"]),
        "test manifest",
    )

    linked_files = (
        ("pretest_selection_report", "pretest_selection_report_sha256"),
        ("raw_benchmark", "raw_benchmark_sha256"),
        ("benchmark_audit", "benchmark_audit_sha256"),
        ("benchmark_protocol", "benchmark_protocol_sha256"),
    )
    for path_key, hash_key in linked_files:
        _require_hash(
            _frozen_path(experiment[path_key]),
            str(experiment[hash_key]),
            path_key.replace("_", " "),
        )

    calibration = experiment.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("source") != "training split only":
        raise ValueError("frozen calibration provenance is missing or invalid")
    if calibration.get("sample_count") != 800:
        raise ValueError("frozen calibration sample count differs from 800")
    _require_hash(
        _frozen_path(calibration["npz"]),
        str(calibration["npz_sha256"]),
        "training-only calibration tensor",
    )

    models = experiment.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("exactly three frozen model specifications are required")
    architectures = {str(model.get("architecture")) for model in models}
    if architectures != EXPECTED_ARCHITECTURES:
        raise ValueError(f"unexpected frozen architecture set: {sorted(architectures)}")
    for model in models:
        architecture = str(model["architecture"])
        _require_hash(
            _frozen_path(model["keras_model"]),
            str(model["keras_sha256"]),
            f"{architecture} Keras model",
        )
        frozen_variants = model.get("validation_variants")
        if not isinstance(frozen_variants, dict) or set(frozen_variants) != set(VARIANTS):
            raise ValueError(f"incomplete frozen variants for {architecture}")
        seen_paths: set[Path] = set()
        for variant in VARIANTS:
            metrics = frozen_variants[variant]
            path = _frozen_path(metrics["path"])
            if _variant_name(path.name) != variant:
                raise ValueError(f"variant/path mismatch for {architecture}/{variant}")
            resolved = path.resolve()
            if resolved in seen_paths:
                raise ValueError(f"duplicate flatbuffer path for {architecture}")
            seen_paths.add(resolved)
            if resolved.parent != _frozen_path(model["tflite_dir"]).resolve():
                raise ValueError(f"flatbuffer outside frozen directory: {path}")
            _require_hash(path, str(metrics["sha256"]), f"{architecture}/{variant}")
    return sorted(models, key=lambda value: str(value["architecture"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--start-record", type=Path)
    args = parser.parse_args()

    start_record = args.start_record or args.report.with_name("final_test_started.json")
    for output in (start_record, args.report, args.predictions):
        if output.exists():
            raise SystemExit(
                f"single-session safeguard: output already exists; refusing rerun: {output}"
            )

    experiment = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = Path(config["dataset"]["manifest_dir"]) / "test.csv"
    frozen_models = _preflight(experiment, config, manifest_path)

    import tensorflow as tf

    gpu_report = require_gpu(tf, args.report.with_suffix(".gpu.json"))
    start_record.parent.mkdir(parents=True, exist_ok=True)
    start_payload = {
        "status": "FINAL_TEST_SESSION_STARTED_DO_NOT_RERUN",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest": str(args.manifest),
        "frozen_manifest_sha256": _sha256(args.manifest),
        "test_manifest_sha256": _sha256(manifest_path),
        "report_target": str(args.report),
        "predictions_target": str(args.predictions),
        "gpu_report": gpu_report,
    }
    with start_record.open("x", encoding="utf-8") as handle:
        json.dump(start_payload, handle, indent=2)
        handle.write("\n")

    image_root = _dataset_root(config["dataset"]["root"]) / "images"
    paths, expected = read_manifest(manifest_path, image_root)
    if len(expected) != int(experiment["test_manifest_record_count"]):
        raise RuntimeError("test manifest record count differs from frozen metadata")
    raw_rows: list[dict[str, object]] = []
    model_results: dict[str, object] = {}

    for model_spec in frozen_models:
        architecture = str(model_spec["architecture"])
        keras_path = _frozen_path(model_spec["keras_model"])
        dataset, labels = build_dataset(
            tf,
            manifest_path,
            image_root,
            1,
            False,
            int(config["project"]["random_seed"]),
        )
        if labels != expected:
            raise RuntimeError("test label order changed during final evaluation")
        keras_model = tf.keras.models.load_model(keras_path)
        runtimes: dict[str, dict[str, object]] = {}
        for variant in VARIANTS:
            metrics = model_spec["validation_variants"][variant]
            path = _frozen_path(metrics["path"])
            interpreter = tf.lite.Interpreter(
                model_path=str(path),
                experimental_op_resolver_type=(
                    tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                ),
            )
            interpreter.allocate_tensors()
            runtimes[variant] = {
                "path": path,
                "interpreter": interpreter,
                "input": interpreter.get_input_details()[0],
                "output": interpreter.get_output_details()[0],
                "predicted": [],
                "confidence": [],
                "agreement": 0,
            }

        keras_predicted: list[int] = []
        keras_confidence: list[float] = []
        for sample_index, (images, _) in enumerate(dataset):
            with tf.device("/GPU:0"):
                reference_tensor = keras_model(images, training=False)
            assert_tensor_on_gpu(reference_tensor)
            reference = reference_tensor.numpy()[0]
            reference_class = int(np.argmax(reference))
            keras_predicted.append(reference_class)
            keras_confidence.append(float(reference[reference_class]))
            float_input = images.numpy()
            for runtime in runtimes.values():
                interpreter = runtime["interpreter"]
                interpreter.set_tensor(
                    runtime["input"]["index"],
                    quantize_input(float_input, runtime["input"]),
                )
                interpreter.invoke()
                probabilities = dequantize_output(
                    interpreter.get_tensor(runtime["output"]["index"]),
                    runtime["output"],
                )[0]
                predicted = int(np.argmax(probabilities))
                runtime["predicted"].append(predicted)
                runtime["confidence"].append(float(probabilities[predicted]))
                runtime["agreement"] += int(predicted == reference_class)
            if (sample_index + 1) % 500 == 0:
                print(f"{architecture}: {sample_index + 1}/{len(expected)}", flush=True)

        variants: dict[str, object] = {
            "keras_reference": {
                "path": str(keras_path),
                "sha256": _sha256(keras_path),
                **_metrics(expected, keras_predicted),
            }
        }
        for sample_index, filename in enumerate(paths):
            raw_rows.append(
                {
                    "sample_index": sample_index,
                    "filename": Path(filename).name,
                    "label": expected[sample_index],
                    "architecture": architecture,
                    "variant": "keras_reference",
                    "predicted": keras_predicted[sample_index],
                    "confidence": f"{keras_confidence[sample_index]:.9f}",
                }
            )
        for name, runtime in runtimes.items():
            predicted = runtime["predicted"]
            variants[name] = {
                "path": str(runtime["path"]),
                "sha256": _sha256(runtime["path"]),
                "top1_agreement_with_keras": runtime["agreement"] / len(expected),
                **_metrics(expected, predicted),
            }
            for sample_index, filename in enumerate(paths):
                raw_rows.append(
                    {
                        "sample_index": sample_index,
                        "filename": Path(filename).name,
                        "label": expected[sample_index],
                        "architecture": architecture,
                        "variant": name,
                        "predicted": predicted[sample_index],
                        "confidence": f"{runtime['confidence'][sample_index]:.9f}",
                    }
                )
        model_results[architecture] = {"variants": variants}
        del keras_model
        tf.keras.backend.clear_session()

    expected_prediction_rows = len(expected) * len(frozen_models) * (len(VARIANTS) + 1)
    if len(raw_rows) != expected_prediction_rows:
        raise RuntimeError("prediction row count differs from the frozen evaluation matrix")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "filename",
                "label",
                "architecture",
                "variant",
                "predicted",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    report = {
        "status": "FINAL_TEST_SINGLE_SESSION_COMPLETE",
        "selection_frozen_before_test": True,
        "single_session_start_record": str(start_record),
        "single_session_start_record_sha256": _sha256(start_record),
        "split": "test",
        "dataset_protocol": config["dataset"]["split_protocol"],
        "test_manifest": str(manifest_path),
        "test_manifest_sha256": _sha256(manifest_path),
        "sample_count": len(expected),
        "keras_execution_device": "/GPU:0",
        "tflite_runtime": "builtin_without_default_delegates_cpu",
        "gpu_report": gpu_report,
        "prediction_file": str(args.predictions),
        "prediction_file_sha256": _sha256(args.predictions),
        "models": model_results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
