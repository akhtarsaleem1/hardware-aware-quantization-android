"""Evaluate a saved Keras model on a non-training split with a required GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tf_pipeline import build_dataset
from src.training.device import assert_tensor_on_gpu, require_gpu


def resolve_dataset_root(raw_value: str) -> Path:
    prefix = "${DATASET_PATH:-"
    if raw_value.startswith(prefix) and raw_value.endswith("}"):
        return Path(os.environ.get("DATASET_PATH", raw_value[len(prefix) : -1]))
    return Path(os.path.expandvars(raw_value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    import tensorflow as tf

    gpu = require_gpu(tf, args.report.with_suffix(".gpu.json"))
    dataset_root = resolve_dataset_root(config["dataset"]["root"])
    dataset, _ = build_dataset(
        tf,
        Path(config["dataset"]["manifest_dir"]) / f"{args.split}.csv",
        dataset_root / "images",
        int(config["training"]["batch_size"]),
        False,
        int(config["project"]["random_seed"]),
    )
    model = tf.keras.models.load_model(args.model)
    sample_images, _ = next(iter(dataset.take(1)))
    with tf.device("/GPU:0"):
        sample_output = model(sample_images, training=False)
    assert_tensor_on_gpu(sample_output)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    metrics = model.evaluate(dataset, return_dict=True, verbose=2)
    report = {
        "status": "VALIDATION_ONLY" if args.split == "validation" else "FINAL_TEST_EVALUATION",
        "model": str(args.model),
        "split": args.split,
        "dataset_protocol": config["dataset"]["split_protocol"],
        "output_device": sample_output.device,
        "gpu_report": gpu,
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
