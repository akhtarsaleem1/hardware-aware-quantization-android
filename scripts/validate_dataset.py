"""Audit DeepWeeds manifests before any model is trained."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from PIL import Image, UnidentifiedImageError


SPLITS = ("train", "validation", "test")
CAPTURE_PATTERN = re.compile(r"^(\d{8})-(\d{4})\d{2}-(\d+)")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(
        gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata()
    )
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def hash_blocks(value: int) -> Iterable[tuple[int, int]]:
    """Five exact blocks guarantee a candidate for Hamming distance <= 4."""

    widths = (13, 13, 13, 13, 12)
    shift = 64
    for index, width in enumerate(widths):
        shift -= width
        yield index, (value >> shift) & ((1 << width) - 1)


def capture_session(filename: str) -> str | None:
    match = CAPTURE_PATTERN.match(Path(filename).stem)
    if not match:
        return None
    day, minute, instrument = match.groups()
    return f"{day}-{minute}-{instrument}"


def load_manifests(directory: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    required = {"filename", "label", "species", "source_split", "source_fold"}
    for split in SPLITS:
        path = directory / f"{split}.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} lacks required columns")
            for row in reader:
                if row["source_split"] != split:
                    raise ValueError(f"{path}: source_split mismatch for {row['filename']}")
                records.append(dict(row))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/dataset_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/dataset_report.md"))
    parser.add_argument(
        "--near-duplicates-csv",
        type=Path,
        default=Path("reports/dataset_near_duplicate_candidates.csv"),
    )
    parser.add_argument(
        "--near-review-csv",
        type=Path,
        default=Path("reports/dataset_near_duplicate_review.csv"),
    )
    parser.add_argument("--near-distance", type=int, default=4, choices=range(0, 9))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.dataset_root.resolve() / "images"
    records = load_manifests(args.manifest_dir)
    filename_owners: dict[str, set[str]] = defaultdict(set)
    digest_owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    session_owners: dict[str, set[str]] = defaultdict(set)
    class_counts: dict[str, Counter[str]] = {name: Counter() for name in SPLITS}
    dimension_counts: Counter[str] = Counter()
    missing: list[str] = []
    unreadable: list[str] = []
    phashes: list[tuple[str, str, int]] = []

    for record in records:
        filename = record["filename"]
        split = record["source_split"]
        filename_owners[filename].add(split)
        class_counts[split][record["label"]] += 1
        session = capture_session(filename)
        if session:
            session_owners[session].add(split)
        image_path = images_dir / filename
        if not image_path.is_file():
            missing.append(filename)
            continue
        digest = record.get("image_sha256") or ""
        if digest in {"", "NOT_COMPUTED", "MISSING"}:
            digest = sha256_file(image_path)
        digest_owners[digest].append((split, filename))
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                dimension_counts[f"{image.width}x{image.height}:{image.mode}"] += 1
                phashes.append((split, filename, difference_hash(image)))
        except (OSError, UnidentifiedImageError) as exc:
            unreadable.append(f"{filename}: {exc}")

    filename_overlap = {
        name: sorted(owners)
        for name, owners in filename_owners.items()
        if len(owners) > 1
    }
    exact_cross_split = []
    for digest, owners in digest_owners.items():
        splits = {split for split, _ in owners}
        if len(splits) > 1:
            exact_cross_split.append(
                {"sha256": digest, "members": owners, "splits": sorted(splits)}
            )

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    near_pairs: list[dict[str, object]] = []
    for index, (split, filename, value) in enumerate(phashes):
        candidates: set[int] = set()
        for block in hash_blocks(value):
            candidates.update(buckets[block])
        for candidate_index in candidates:
            other_split, other_filename, other_value = phashes[candidate_index]
            if split == other_split:
                continue
            distance = (value ^ other_value).bit_count()
            if 0 < distance <= args.near_distance:
                near_pairs.append(
                    {
                        "filename_a": other_filename,
                        "split_a": other_split,
                        "filename_b": filename,
                        "split_b": split,
                        "dhash_distance": distance,
                    }
                )
        for block in hash_blocks(value):
            buckets[block].append(index)

    cross_split_sessions = {
        key: sorted(owners)
        for key, owners in session_owners.items()
        if len(owners) > 1
    }
    reviewed: dict[tuple[str, str], dict[str, str]] = {}
    if args.near_review_csv.is_file():
        with args.near_review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = tuple(sorted((row["filename_a"], row["filename_b"])))
                reviewed[key] = row
    unresolved_near = []
    merge_required = []
    reviewed_distinct = []
    for pair in near_pairs:
        key = tuple(sorted((str(pair["filename_a"]), str(pair["filename_b"]))))
        decision = reviewed.get(key, {}).get("decision", "").strip()
        if decision == "distinct_scene":
            reviewed_distinct.append(pair)
        elif decision == "merge_required":
            merge_required.append(pair)
        else:
            unresolved_near.append(pair)
    hard_failure = bool(
        missing or unreadable or filename_overlap or exact_cross_split or merge_required
    )
    needs_review = bool(unresolved_near or cross_split_sessions)
    report = {
        "status": "FAIL" if hard_failure else "REVIEW_REQUIRED" if needs_review else "PASS",
        "record_count": len(records),
        "split_counts": dict(Counter(row["source_split"] for row in records)),
        "class_counts": {key: dict(sorted(value.items())) for key, value in class_counts.items()},
        "image_dimensions": dict(dimension_counts),
        "missing_count": len(missing),
        "unreadable_count": len(unreadable),
        "filename_cross_split_count": len(filename_overlap),
        "exact_duplicate_cross_split_group_count": len(exact_cross_split),
        "near_duplicate_cross_split_candidate_count": len(near_pairs),
        "near_duplicate_reviewed_distinct_count": len(reviewed_distinct),
        "near_duplicate_unresolved_count": len(unresolved_near),
        "near_duplicate_merge_required_count": len(merge_required),
        "capture_minute_instrument_cross_split_count": len(cross_split_sessions),
        "near_duplicate_distance_threshold": args.near_distance,
        "notes": [
            "Perceptual-hash matches are candidates requiring visual review, not proven duplicates.",
            "Capture-session key is filename date + minute + instrument and is a leakage-risk heuristic.",
        ],
        "failures": {
            "missing": missing,
            "unreadable": unreadable,
            "filename_overlap": filename_overlap,
            "exact_cross_split": exact_cross_split,
            "near_duplicate_merge_required": merge_required,
            "near_duplicate_unresolved": unresolved_near,
        },
    }

    for path in (args.report_json, args.report_md, args.near_duplicates_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.near_duplicates_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["filename_a", "split_a", "filename_b", "split_b", "dhash_distance"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(near_pairs)

    markdown = f"""# DeepWeeds dataset validation report

Status: **{report['status']}**

This file is generated by `scripts/validate_dataset.py`; do not replace its
counts with estimates.

| Check | Observed |
|---|---:|
| Manifest records | {report['record_count']} |
| Missing images | {report['missing_count']} |
| Unreadable images | {report['unreadable_count']} |
| Filename cross-split overlaps | {report['filename_cross_split_count']} |
| Exact-hash cross-split groups | {report['exact_duplicate_cross_split_group_count']} |
| Near-duplicate cross-split candidates | {report['near_duplicate_cross_split_candidate_count']} |
| Near-duplicate candidates reviewed as distinct | {report['near_duplicate_reviewed_distinct_count']} |
| Near-duplicate candidates unresolved | {report['near_duplicate_unresolved_count']} |
| Capture-minute/instrument groups crossing splits | {report['capture_minute_instrument_cross_split_count']} |

Near-duplicate candidates and capture-session warnings require review before
the split is frozen. A `PASS` status does not establish external validity; it
only means these implemented integrity checks did not detect a blocker.
"""
    args.report_md.write_text(markdown, encoding="utf-8")
    metadata_path = args.manifest_dir / "manifest_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["validation_status"] = report["status"]
        metadata["validation_report"] = str(args.report_json)
        if report["status"] == "PASS":
            metadata["status"] = "FROZEN_PRETRAINING_SPLIT"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "failures"}, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
