"""Run a one-batch GPU feasibility gate for every selected backbone."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tf_pipeline import build_dataset
from src.models.factory import SUPPORTED_MODELS, build_classifier
from src.training.device import assert_tensor_on_gpu, require_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/model_feasibility.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["project"]["random_seed"])
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    gpu_report = require_gpu(tf, "environment/model_feasibility_gpu_report.json")
    manifest_dir = Path(config["dataset"]["manifest_dir"])
    dataset, _ = build_dataset(
        tf,
        manifest_dir / "train.csv",
        args.dataset_root / "images",
        args.batch_size,
        True,
        seed,
    )
    images, labels = next(iter(dataset.take(1)))
    outcomes = []
    for model_name in SUPPORTED_MODELS:
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            model, backbone = build_classifier(tf, model_name)
            backbone.trainable = True
            for layer in backbone.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
            optimizer = tf.keras.optimizers.Adam(
                float(config["training"]["fine_tuning_learning_rate"])
            )
            with tf.device("/GPU:0"), tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        labels, predictions
                    )
                )
            gradients = tape.gradient(loss, model.trainable_variables)
            gradient_pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, model.trainable_variables)
                if gradient is not None
            ]
            if not gradient_pairs:
                raise RuntimeError("no trainable gradients")
            optimizer.apply_gradients(gradient_pairs)
            assert_tensor_on_gpu(predictions)
            memory = tf.config.experimental.get_memory_info("GPU:0")
            outcomes.append(
                {
                    "model": model_name,
                    "status": "PASS",
                    "batch_size": args.batch_size,
                    "parameter_count": model.count_params(),
                    "trainable_variable_count": len(model.trainable_variables),
                    "output_shape": predictions.shape.as_list(),
                    "output_device": predictions.device,
                    "smoke_loss_not_a_result": float(loss.numpy()),
                    "gpu_memory": memory,
                }
            )
        except Exception as exc:
            outcomes.append(
                {
                    "model": model_name,
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    report = {
        "status": "PASS" if all(row["status"] == "PASS" for row in outcomes) else "FAIL",
        "purpose": "one-batch full-backbone fine-tuning feasibility only; no accuracy claim",
        "dataset_split": config["dataset"]["split_protocol"],
        "gpu_report": gpu_report,
        "models": outcomes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
