"""Create canonical manifests from one official DeepWeeds fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deepweeds import load_official_split, validate_disjoint_splits


SPLIT_FILES = {
    "train": "train_subset{fold}.csv",
    "validation": "val_subset{fold}.csv",
    "test": "test_subset{fold}.csv",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--hash-images",
        action="store_true",
        help="Compute per-image SHA-256 values (slower but required before training).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    labels_dir = source / "labels"
    images_dir = source / "images"
    if not labels_dir.is_dir() or not images_dir.is_dir():
        raise SystemExit("source root must contain labels/ and images/")

    splits = {
        name: load_official_split(labels_dir / pattern.format(fold=args.fold))
        for name, pattern in SPLIT_FILES.items()
    }
    validate_disjoint_splits(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    manifest_hashes: dict[str, str] = {}
    for split_name, rows in splits.items():
        output_path = args.output_dir / f"{split_name}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "filename",
                    "label",
                    "species",
                    "source_split",
                    "source_fold",
                    "image_sha256",
                ],
            )
            writer.writeheader()
            for row in rows:
                image_path = images_dir / row.filename
                if not image_path.is_file():
                    missing.append(row.filename)
                    image_hash = "MISSING"
                else:
                    image_hash = sha256_file(image_path) if args.hash_images else "NOT_COMPUTED"
                writer.writerow(
                    {
                        "filename": row.filename,
                        "label": row.label,
                        "species": row.species,
                        "source_split": split_name,
                        "source_fold": args.fold,
                        "image_sha256": image_hash,
                    }
                )
        manifest_hashes[split_name] = sha256_file(output_path)

    metadata = {
        "dataset": "DeepWeeds",
        "source_root": str(source),
        "official_fold": args.fold,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "hash_images": args.hash_images,
        "missing_image_count": len(missing),
        "manifest_sha256": manifest_hashes,
    }
    (args.output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if missing:
        raise SystemExit(f"{len(missing)} manifest images are missing; first: {missing[:5]}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
