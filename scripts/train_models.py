"""Two-stage, fail-closed GPU training for the frozen DeepWeeds split."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
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
    parser.add_argument("--model", choices=(*SUPPORTED_MODELS, "all"), default="all")
    return parser.parse_args()


def balanced_class_weights(labels: list[int]) -> dict[int, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    return {label: total / (len(counts) * count) for label, count in counts.items()}


def resolve_dataset_root(raw_value: str) -> Path:
    prefix = "${DATASET_PATH:-"
    if raw_value.startswith(prefix) and raw_value.endswith("}"):
        default = raw_value[len(prefix) : -1]
        return Path(os.environ.get("DATASET_PATH", default))
    return Path(os.path.expandvars(raw_value))


def compile_model(tf, model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["project"]["random_seed"])
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    device_report_path = f"environment/training_gpu_report_{run_id}.json"
    device_report = require_gpu(tf, device_report_path)
    if config["training"].get("mixed_precision", False):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    dataset_root = resolve_dataset_root(config["dataset"]["root"])
    manifest_dir = Path(config["dataset"]["manifest_dir"])
    image_dir = dataset_root / "images"
    batch_size = int(config["training"]["batch_size"])
    train_data, train_labels = build_dataset(
        tf, manifest_dir / "train.csv", image_dir, batch_size, True, seed
    )
    validation_data, _ = build_dataset(
        tf, manifest_dir / "validation.csv", image_dir, batch_size, False, seed
    )
    weights = balanced_class_weights(train_labels)
    selected = SUPPORTED_MODELS if args.model == "all" else (args.model,)
    for model_name in selected:
        tf.keras.backend.clear_session()
        model, backbone = build_classifier(tf, model_name)
        backbone.trainable = False
        sample_images, sample_labels = next(iter(train_data.take(1)))
        with tf.device("/GPU:0"), tf.GradientTape() as tape:
            prediction = model(sample_images, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(sample_labels, prediction)
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        if not any(gradient is not None for gradient in gradients):
            raise RuntimeError("GPU smoke test produced no gradients")
        assert_tensor_on_gpu(prediction)

        checkpoint_dir = Path("models/checkpoints") / model_name / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        def callbacks(stage: str):
            return [
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_dir / f"best_{stage}.keras",
                    monitor="val_loss",
                    save_best_only=True,
                ),
                tf.keras.callbacks.CSVLogger(checkpoint_dir / f"{stage}_training_log.csv"),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=int(config["training"]["early_stopping_patience"]),
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
                ),
            ]
        frozen_epochs = int(config["training"]["frozen_backbone_epochs"])
        compile_model(tf, model, float(config["training"]["learning_rate"]))
        frozen_history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=frozen_epochs,
            class_weight=weights,
            callbacks=callbacks("frozen"),
            verbose=2,
        )
        backbone.trainable = True
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        compile_model(tf, model, float(config["training"]["fine_tuning_learning_rate"]))
        fine_history = model.fit(
            train_data,
            validation_data=validation_data,
            initial_epoch=frozen_epochs,
            epochs=int(config["training"]["epochs"]),
            class_weight=weights,
            callbacks=callbacks("fine_tuning"),
            verbose=2,
        )
        saved_dir = Path("models/saved") / model_name
        metadata_dir = Path("models/metadata") / model_name
        saved_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        model_path = saved_dir / f"{run_id}.keras"
        model.save(model_path)
        metadata = {
            "run_id": run_id,
            "model": model_name,
            "model_path": str(model_path),
            "parameter_count": model.count_params(),
            "trainable_parameter_count": sum(
                int(np.prod(variable.shape)) for variable in model.trainable_variables
            ),
            "class_weights": weights,
            "seed": seed,
            "dataset_split": config["dataset"]["split_protocol"],
            "gpu_report_path": device_report_path,
            "gpu_report": device_report,
            "frozen_history": frozen_history.history,
            "fine_tuning_history": fine_history.history,
            "status": "TRAINED_NOT_YET_TEST_EVALUATED",
        }
        (metadata_dir / f"{run_id}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
