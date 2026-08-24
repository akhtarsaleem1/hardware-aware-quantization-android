"""Resume an interrupted single-model fine-tuning run on a required GPU."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
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
from src.models.factory import SUPPORTED_MODELS
from src.training.device import assert_tensor_on_gpu, require_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def resolve_dataset_root(raw_value: str) -> Path:
    prefix = "${DATASET_PATH:-"
    if raw_value.startswith(prefix) and raw_value.endswith("}"):
        default = raw_value[len(prefix) : -1]
        return Path(os.environ.get("DATASET_PATH", default))
    return Path(os.path.expandvars(raw_value))


def balanced_class_weights(labels: list[int]) -> dict[int, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    return {label: total / (len(counts) * count) for label, count in counts.items()}


def completed_epoch_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        epochs = [int(row["epoch"]) for row in csv.DictReader(handle)]
    return max(epochs, default=-1) + 1


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["project"]["random_seed"])
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    run_id = args.checkpoint.resolve().parent.name
    device_report_path = f"environment/training_gpu_report_{run_id}_resume.json"
    device_report = require_gpu(tf, device_report_path)

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

    model = tf.keras.models.load_model(args.checkpoint)
    sample_images, sample_labels = next(iter(train_data.take(1)))
    with tf.device("/GPU:0"), tf.GradientTape() as tape:
        prediction = model(sample_images, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(sample_labels, prediction)
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    if not any(gradient is not None for gradient in gradients):
        raise RuntimeError("GPU resume smoke test produced no gradients")
    assert_tensor_on_gpu(prediction)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            float(config["training"]["fine_tuning_learning_rate"])
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    checkpoint_dir = args.checkpoint.resolve().parent
    fine_log_path = checkpoint_dir / "fine_tuning_training_log.csv"
    initial_epoch = completed_epoch_count(fine_log_path)
    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=checkpoint_dir / "backup_fine_tuning",
            delete_checkpoint=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_dir / "best_fine_tuning.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(fine_log_path, append=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config["training"]["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
        ),
    ]
    history = model.fit(
        train_data,
        validation_data=validation_data,
        initial_epoch=initial_epoch,
        epochs=int(config["training"]["epochs"]),
        class_weight=balanced_class_weights(train_labels),
        callbacks=callbacks,
        verbose=2,
    )

    saved_dir = Path("models/saved") / args.model
    metadata_dir = Path("models/metadata") / args.model
    saved_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    model_path = saved_dir / f"{run_id}.keras"
    model.save(model_path)
    metadata = {
        "run_id": run_id,
        "model": args.model,
        "model_path": str(model_path),
        "parameter_count": model.count_params(),
        "seed": seed,
        "dataset_split": config["dataset"]["split_protocol"],
        "gpu_report_path": device_report_path,
        "gpu_report": device_report,
        "resumed_from": str(args.checkpoint),
        "resume_initial_epoch": initial_epoch,
        "resume_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "fine_tuning_history_after_resume": history.history,
        "full_history_csv": str(fine_log_path),
        "status": "TRAINED_NOT_YET_TEST_EVALUATED",
    }
    (metadata_dir / f"{run_id}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
