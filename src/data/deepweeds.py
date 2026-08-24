"""Strict parsing helpers for the official DeepWeeds CSV split files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CLASS_NAMES = {
    0: "Chinee Apple",
    1: "Lantana",
    2: "Parkinsonia",
    3: "Parthenium",
    4: "Prickly Acacia",
    5: "Rubber Vine",
    6: "Siam Weed",
    7: "Snake Weed",
    8: "Negative",
}


@dataclass(frozen=True)
class DeepWeedsRow:
    filename: str
    label: int
    species: str


def load_official_split(path: str | Path) -> list[DeepWeedsRow]:
    """Read an official labels/split CSV and reject ambiguous records."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Filename", "Label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain columns {sorted(required)}")
        rows: list[DeepWeedsRow] = []
        seen: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            filename = (raw.get("Filename") or "").strip()
            try:
                label = int((raw.get("Label") or "").strip())
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{line_number}: invalid Label") from exc
            if not filename or Path(filename).name != filename:
                raise ValueError(f"{csv_path}:{line_number}: unsafe Filename")
            if label not in CLASS_NAMES:
                raise ValueError(f"{csv_path}:{line_number}: label {label} outside 0..8")
            # labels.csv includes Species, while the official fold CSVs contain
            # only Filename and Label despite the repository README saying the
            # split files use the same format. Use the fixed published map.
            species = (raw.get("Species") or CLASS_NAMES[label]).strip()
            if filename in seen:
                raise ValueError(f"{csv_path}:{line_number}: duplicate {filename}")
            seen.add(filename)
            rows.append(DeepWeedsRow(filename, label, species))
    if not rows:
        raise ValueError(f"{csv_path} contains no records")
    return rows


def validate_disjoint_splits(splits: dict[str, Iterable[DeepWeedsRow]]) -> None:
    """Reject filenames assigned to more than one split."""

    owner: dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            previous = owner.setdefault(row.filename, split_name)
            if previous != split_name:
                raise ValueError(
                    f"{row.filename} occurs in both {previous} and {split_name}"
                )
