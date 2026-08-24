"""Compare TFLite variants with a Keras reference on the frozen validation split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tf_pipeline import build_dataset
from src.quantization.tensor_io import dequantize_output, quantize_input
from src.training.device import assert_tensor_on_gpu, require_gpu


def resolve_dataset_root(raw_value: str) -> Path:
    prefix = "${DATASET_PATH:-"
    if raw_value.startswith(prefix) and raw_value.endswith("}"):
        return Path(os.environ.get("DATASET_PATH", raw_value[len(prefix) : -1]))
    return Path(os.path.expandvars(raw_value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras-model", type=Path, required=True)
    parser.add_argument("--tflite-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    import tensorflow as tf

    gpu_report = require_gpu(tf, args.report.with_suffix(".gpu.json"))
    dataset_root = resolve_dataset_root(config["dataset"]["root"])
    dataset, labels = build_dataset(
        tf,
        Path(config["dataset"]["manifest_dir"]) / "validation.csv",
        dataset_root / "images",
        1,
        False,
        int(config["project"]["random_seed"]),
    )
    keras_model = tf.keras.models.load_model(args.keras_model)

    runtimes = {}
    for path in sorted(args.tflite_dir.glob("*.tflite")):
        interpreter = tf.lite.Interpreter(
            model_path=str(path),
            experimental_op_resolver_type=(
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        )
        interpreter.allocate_tensors()
        runtimes[path.stem] = {
            "path": str(path),
            "interpreter": interpreter,
            "input": interpreter.get_input_details()[0],
            "output": interpreter.get_output_details()[0],
            "correct": 0,
            "agreement": 0,
            "absolute_probability_error_sum": 0.0,
            "maximum_absolute_probability_error": 0.0,
        }

    keras_correct = 0
    observed = 0
    for images, batch_labels in dataset:
        with tf.device("/GPU:0"):
            keras_probabilities_tensor = keras_model(images, training=False)
        assert_tensor_on_gpu(keras_probabilities_tensor)
        keras_probabilities = keras_probabilities_tensor.numpy()[0]
        label = int(batch_labels.numpy()[0])
        keras_class = int(np.argmax(keras_probabilities))
        keras_correct += int(keras_class == label)
        float_input = images.numpy()

        for runtime in runtimes.values():
            interpreter = runtime["interpreter"]
            interpreter.set_tensor(
                runtime["input"]["index"], quantize_input(float_input, runtime["input"])
            )
            interpreter.invoke()
            probabilities = dequantize_output(
                interpreter.get_tensor(runtime["output"]["index"]), runtime["output"]
            )[0]
            predicted = int(np.argmax(probabilities))
            absolute_error = np.abs(probabilities - keras_probabilities)
            runtime["correct"] += int(predicted == label)
            runtime["agreement"] += int(predicted == keras_class)
            runtime["absolute_probability_error_sum"] += float(absolute_error.sum())
            runtime["maximum_absolute_probability_error"] = max(
                runtime["maximum_absolute_probability_error"], float(absolute_error.max())
            )
        observed += 1
        if observed % 500 == 0:
            print(f"processed {observed}/{len(labels)}", flush=True)

    variant_results = {}
    class_count = 9
    for name, runtime in runtimes.items():
        variant_results[name] = {
            "path": runtime["path"],
            "sample_count": observed,
            "accuracy": runtime["correct"] / observed,
            "top1_agreement_with_keras": runtime["agreement"] / observed,
            "mean_absolute_probability_error": (
                runtime["absolute_probability_error_sum"] / (observed * class_count)
            ),
            "maximum_absolute_probability_error": runtime[
                "maximum_absolute_probability_error"
            ],
        }
    report = {
        "status": "VALIDATION_PARITY_ONLY_TEST_SPLIT_UNTOUCHED",
        "split": "validation",
        "dataset_protocol": config["dataset"]["split_protocol"],
        "sample_count": observed,
        "keras_model": str(args.keras_model),
        "keras_reference_accuracy": keras_correct / observed,
        "keras_execution_device": "/GPU:0",
        "tflite_runtime": "builtin_without_default_delegates_cpu",
        "gpu_report": gpu_report,
        "variants": variant_results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
