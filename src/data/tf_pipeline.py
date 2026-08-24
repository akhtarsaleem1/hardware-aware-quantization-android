"""TensorFlow input pipeline backed by frozen CSV manifests."""

from __future__ import annotations

import csv
from pathlib import Path


def read_manifest(path: str | Path, image_directory: str | Path):
    image_root = Path(image_directory)
    paths: list[str] = []
    labels: list[int] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            paths.append(str(image_root / row["filename"]))
            labels.append(int(row["label"]))
    if not paths:
        raise ValueError(f"empty manifest: {path}")
    return paths, labels


def build_dataset(
    tf,
    manifest_path: str | Path,
    image_directory: str | Path,
    batch_size: int,
    training: bool,
    seed: int,
):
    paths, labels = read_manifest(manifest_path, image_directory)
    indices = list(range(len(paths)))
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, indices))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label, index):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32) * 255.0
        image = tf.image.resize(image, (256, 256), antialias=True)
        if training:
            base_seed = tf.stack((tf.cast(seed, tf.int32), tf.cast(index, tf.int32)))
            image = tf.image.stateless_random_crop(image, (224, 224, 3), seed=base_seed)
            image = tf.image.stateless_random_flip_left_right(
                image, seed=base_seed + tf.constant((0, 1), tf.int32)
            )
            image = tf.image.stateless_random_brightness(
                image, max_delta=20.0, seed=base_seed + tf.constant((0, 2), tf.int32)
            )
            image = tf.image.stateless_random_contrast(
                image, 0.85, 1.15, seed=base_seed + tf.constant((0, 3), tf.int32)
            )
            image = tf.clip_by_value(image, 0.0, 255.0)
        else:
            image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        return image, label

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE), labels
