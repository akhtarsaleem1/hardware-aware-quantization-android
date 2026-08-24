"""Create a deterministic, class-stratified calibration tensor from training only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests_grouped/train.csv"))
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(int(row["label"]), []).append(row)
    if args.samples < len(groups):
        raise ValueError("sample count must cover every training class")

    rng = random.Random(args.seed)
    for group in groups.values():
        rng.shuffle(group)
    base, remainder = divmod(args.samples, len(groups))
    selected: list[dict[str, str]] = []
    for position, label in enumerate(sorted(groups)):
        count = base + (1 if position < remainder else 0)
        if len(groups[label]) < count:
            raise ValueError(f"class {label} has fewer than {count} training samples")
        selected.extend(groups[label][:count])
    rng.shuffle(selected)

    inputs = np.empty((len(selected), 224, 224, 3), dtype=np.float32)
    filenames: list[str] = []
    labels: list[int] = []
    for index, row in enumerate(selected):
        path = args.image_dir / row["filename"]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            left = (256 - 224) // 2
            array = np.asarray(image.crop((left, left, left + 224, left + 224)), dtype=np.float32)
        inputs[index] = array
        filenames.append(row["filename"])
        labels.append(int(row["label"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        inputs=inputs,
        filenames=np.asarray(filenames),
        labels=np.asarray(labels, dtype=np.int32),
    )
    metadata = {
        "status": "CALIBRATION_ONLY_NOT_EVALUATION",
        "source_manifest": str(args.manifest),
        "source_split": "train",
        "selection": "deterministic_equal_per_class_without_replacement",
        "seed": args.seed,
        "sample_count": len(selected),
        "class_counts": {
            str(label): labels.count(label) for label in sorted(set(labels))
        },
        "input_shape": list(inputs.shape),
        "input_dtype": str(inputs.dtype),
        "input_range": [float(inputs.min()), float(inputs.max())],
        "npz_path": str(args.output),
        "npz_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
