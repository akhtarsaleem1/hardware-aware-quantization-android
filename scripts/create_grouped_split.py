"""Create a deterministic DeepWeeds split grouped by capture session."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deepweeds import load_official_split


FILENAME_PATTERN = re.compile(r"^(\d{8}-\d{6})-(\d+)\.(?:jpg|jpeg)$", re.I)
SPLIT_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.20}


@dataclass
class CaptureGroup:
    group_id: str
    rows: list

    @property
    def counts(self) -> Counter[int]:
        return Counter(row.label for row in self.rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_capture(filename: str) -> tuple[str, datetime]:
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"unexpected DeepWeeds filename: {filename}")
    timestamp, instrument = match.groups()
    return instrument, datetime.strptime(timestamp, "%Y%m%d-%H%M%S")


def build_groups(rows: list, gap_seconds: int) -> list[CaptureGroup]:
    by_instrument: dict[str, list[tuple[datetime, object]]] = {}
    for row in rows:
        instrument, timestamp = parse_capture(row.filename)
        by_instrument.setdefault(instrument, []).append((timestamp, row))
    groups: list[CaptureGroup] = []
    for instrument, sequence in sorted(by_instrument.items()):
        sequence.sort(key=lambda item: item[0])
        current: list = []
        previous: datetime | None = None
        start: datetime | None = None
        for timestamp, row in sequence:
            if previous is None or (timestamp - previous).total_seconds() <= gap_seconds:
                if not current:
                    start = timestamp
                current.append(row)
            else:
                groups.append(CaptureGroup(f"i{instrument}_{start:%Y%m%dT%H%M%S}", current))
                current = [row]
                start = timestamp
            previous = timestamp
        if current:
            groups.append(CaptureGroup(f"i{instrument}_{start:%Y%m%dT%H%M%S}", current))
    return groups


def assignment_score(
    counts: dict[str, Counter[int]],
    sizes: Counter[str],
    target_counts: dict[str, dict[int, float]],
    target_sizes: dict[str, float],
) -> float:
    class_error = 0.0
    size_error = 0.0
    for split in SPLIT_RATIOS:
        for label, target in target_counts[split].items():
            class_error += ((counts[split][label] - target) / max(target, 1.0)) ** 2
        size_error += ((sizes[split] - target_sizes[split]) / target_sizes[split]) ** 2
    return class_error + 0.25 * size_error


def assign_groups(groups: list[CaptureGroup], seed: int) -> dict[str, str]:
    totals = Counter(row.label for group in groups for row in group.rows)
    total_size = sum(totals.values())
    target_counts = {
        split: {label: count * ratio for label, count in totals.items()}
        for split, ratio in SPLIT_RATIOS.items()
    }
    target_sizes = {split: total_size * ratio for split, ratio in SPLIT_RATIOS.items()}
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda group: (
            max(group.counts[label] / totals[label] for label in group.counts),
            len(group.rows),
        ),
        reverse=True,
    )
    counts = {split: Counter() for split in SPLIT_RATIOS}
    sizes: Counter[str] = Counter()
    assignments: dict[str, str] = {}
    for group in ordered:
        candidates = []
        for split in SPLIT_RATIOS:
            counts[split].update(group.counts)
            sizes[split] += len(group.rows)
            score = assignment_score(counts, sizes, target_counts, target_sizes)
            overflow = max(0.0, sizes[split] - target_sizes[split]) / target_sizes[split]
            candidates.append((score + 10.0 * overflow**2, rng.random(), split))
            counts[split].subtract(group.counts)
            sizes[split] -= len(group.rows)
        _, _, selected = min(candidates)
        assignments[group.group_id] = selected
        counts[selected].update(group.counts)
        sizes[selected] += len(group.rows)
    return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests_grouped"))
    parser.add_argument("--hash-manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--gap-seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_official_split(args.source_root / "labels" / "labels.csv")
    images_dir = args.source_root / "images"
    groups = build_groups(labels, args.gap_seconds)
    assignments = assign_groups(groups, args.seed)
    known_hashes: dict[str, str] = {}
    for split in SPLIT_RATIOS:
        manifest = args.hash_manifest_dir / f"{split}.csv"
        if manifest.is_file():
            with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
                for record in csv.DictReader(handle):
                    value = record.get("image_sha256", "")
                    if value not in {"", "NOT_COMPUTED", "MISSING"}:
                        known_hashes[record["filename"]] = value
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (args.output_dir / f"{split}.csv").open("w", encoding="utf-8", newline="")
        for split in SPLIT_RATIOS
    }
    fieldnames = [
        "filename",
        "label",
        "species",
        "source_split",
        "source_fold",
        "capture_group",
        "image_sha256",
    ]
    writers = {split: csv.DictWriter(handle, fieldnames=fieldnames) for split, handle in handles.items()}
    for writer in writers.values():
        writer.writeheader()
    split_counts = Counter()
    class_counts = {split: Counter() for split in SPLIT_RATIOS}
    try:
        for group in groups:
            split = assignments[group.group_id]
            for row in group.rows:
                image_path = images_dir / row.filename
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                writers[split].writerow(
                    {
                        "filename": row.filename,
                        "label": row.label,
                        "species": row.species,
                        "source_split": split,
                        "source_fold": "grouped_capture_session_v1",
                        "capture_group": group.group_id,
                        "image_sha256": known_hashes.get(row.filename) or file_sha256(image_path),
                    }
                )
                split_counts[split] += 1
                class_counts[split][row.label] += 1
    finally:
        for handle in handles.values():
            handle.close()
    metadata = {
        "dataset": "DeepWeeds",
        "method": "deterministic greedy class-balanced capture-session grouping",
        "seed": args.seed,
        "session_definition": f"same instrument; consecutive timestamp gap <= {args.gap_seconds} seconds",
        "capture_group_count": len(groups),
        "largest_capture_group": max(len(group.rows) for group in groups),
        "split_counts": dict(split_counts),
        "class_counts": {split: dict(sorted(counts.items())) for split, counts in class_counts.items()},
        "manifest_sha256": {
            split: file_sha256(args.output_dir / f"{split}.csv") for split in SPLIT_RATIOS
        },
        "status": "PROVISIONAL_PENDING_DUPLICATE_AUDIT",
    }
    (args.output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
