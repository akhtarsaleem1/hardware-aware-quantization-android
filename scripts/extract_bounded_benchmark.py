#!/usr/bin/env python3
"""Freeze a balanced two-trial exploratory matrix from an interrupted run.

The source artifact remains immutable and excluded from confirmatory analysis.
Only trials that completed for every planned configuration are retained.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--app-manifest", required=True, type=Path)
    parser.add_argument("--trials", default=2, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--failure-report", required=True, type=Path)
    args = parser.parse_args()
    if args.trials != 2:
        raise ValueError("this recovery protocol is frozen to trials 1 and 2")

    manifest = json.loads(args.app_manifest.read_text(encoding="utf-8"))
    models = {str(item["id"]): item for item in manifest["models"]}
    expected_keys = {
        (model_id, runtime, int(threads), trial)
        for model_id in models
        for runtime in manifest["runtimes"]
        for threads in manifest["threads"]
        for trial in range(1, args.trials + 1)
    }
    observations = defaultdict(list)
    errors = defaultdict(list)
    orders = defaultdict(set)
    source_rows = 0
    source_trials = set()
    selected = []
    last_selected_timestamp = ""

    with args.source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("source CSV has no header")
        for line, row in enumerate(reader, start=2):
            source_rows += 1
            trial = int(row["trial_id"])
            source_trials.add(trial)
            if trial > args.trials:
                continue
            model_id = row["model_id"]
            if model_id not in models:
                raise ValueError(f"row {line}: unknown model")
            key = (model_id, row["runtime"], int(row["threads"]), trial)
            orders[key].add(int(row["randomized_order_index"]))
            if row["phase"] == "measured":
                observations[key].append(int(row["run_index"]))
            elif row["phase"] == "configuration_error":
                errors[key].append(row["error"])
            else:
                raise ValueError(f"row {line}: unsupported phase")
            selected.append(row)
            last_selected_timestamp = row["timestamp_utc"]

    if source_trials != {1, 2, 3}:
        raise ValueError(f"source trial coverage is unexpected: {source_trials}")
    if set(observations) | set(errors) != expected_keys:
        raise ValueError("trials 1-2 do not cover the full planned matrix")

    successful = failed = 0
    for model_id in models:
        for runtime in manifest["runtimes"]:
            for threads in manifest["threads"]:
                keys = [
                    (model_id, runtime, int(threads), trial)
                    for trial in range(1, args.trials + 1)
                ]
                if all(
                    sorted(observations[key]) == list(range(100)) and not errors[key]
                    for key in keys
                ):
                    successful += 1
                elif all(not observations[key] and len(errors[key]) == 1 for key in keys):
                    failed += 1
                else:
                    raise ValueError(f"unbalanced trials 1-2: {model_id}/{runtime}/{threads}t")
                if any(len(orders[key]) != 1 for key in keys):
                    raise ValueError(f"order index changed: {model_id}/{runtime}/{threads}t")
    if successful + failed != 72:
        raise ValueError("bounded matrix does not account for 72 configurations")
    for trial in range(1, args.trials + 1):
        trial_orders = {
            next(iter(orders[key])) for key in expected_keys if key[3] == trial
        }
        if trial_orders != set(range(72)):
            raise ValueError(f"trial {trial} order is not a permutation of 0..71")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    failure = {
        "status": "FAILED_INCOMPLETE_EXCLUDED_FROM_CONFIRMATORY_ANALYSIS",
        "reason": "process absent and no terminal completion status; trial 3 stopped after 60/100 measured invocations for mobilenet_v2_float32/xnnpack_cpu/2t",
        "source_raw": str(args.source),
        "source_raw_sha256": sha256(args.source),
        "source_data_rows": source_rows,
        "planned_complete_trials": 3,
        "completed_balanced_trials": 2,
        "partial_trial_excluded_wholesale": 3,
        "pooling_or_imputation": False,
        "confirmatory_android_matrix_completed": False,
    }
    args.failure_report.parent.mkdir(parents=True, exist_ok=True)
    args.failure_report.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")

    status = {
        "status": "BOUNDED_EXPLORATORY_COMPLETE",
        "finished_utc": last_selected_timestamp,
        "raw_csv": str(args.output),
        "raw_csv_sha256": sha256(args.output),
        "included_trials": [1, 2],
        "planned_trials": 3,
        "successful_configurations": successful,
        "failed_configurations": failed,
        "measured_rows": successful * args.trials * 100,
        "configuration_error_rows": failed * args.trials,
        "source_failure_report": str(args.failure_report),
        "source_failure_report_sha256": sha256(args.failure_report),
        "scope": "post_hoc_bounded_exploratory_not_confirmatory",
    }
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
