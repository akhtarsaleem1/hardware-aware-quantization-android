"""Build grouped latency summaries from immutable raw CSV rows."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.statistics import summarize_latencies


GROUP_COLUMNS = [
    "device_id",
    "model_id",
    "model_sha256",
    "architecture",
    "quantization",
    "runtime",
    "requested_delegate",
    "effective_delegate",
    "threads",
    "trial_id",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    configuration_errors = 0
    with args.raw.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if row.get("phase") == "configuration_error":
                configuration_errors += 1
                continue
            if row.get("phase") != "measured":
                raise ValueError(
                    f"Raw row {line_number} has unsupported phase {row.get('phase')!r}"
                )
            missing = [name for name in GROUP_COLUMNS + ["latency_ms"] if not row.get(name)]
            if missing:
                raise ValueError(f"Raw row {line_number} lacks {missing}")
            groups[tuple(row[name] for name in GROUP_COLUMNS)].append(
                float(row["latency_ms"])
            )
    if not groups:
        raise ValueError("Raw benchmark file contains no measured observations")

    rows = []
    for key, values in sorted(groups.items()):
        rows.append(dict(zip(GROUP_COLUMNS, key)) | summarize_latencies(values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} grouped summaries; "
        f"preserved {configuration_errors} configuration errors in raw input"
    )


if __name__ == "__main__":
    main()
