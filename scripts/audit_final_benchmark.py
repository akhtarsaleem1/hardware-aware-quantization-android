#!/usr/bin/env python3
"""Fail-closed independent audit of the completed Android benchmark artifact."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path


SCHEMA = [
    "timestamp_utc", "protocol_version", "device_id", "app_version",
    "build_mode", "model_id", "model_sha256", "architecture",
    "quantization", "input_dtype", "output_dtype", "input_shape",
    "runtime", "requested_delegate", "effective_delegate", "delegate_error",
    "threads", "trial_id", "randomized_order_index", "phase", "run_index",
    "latency_ms", "model_load_ms", "process_pss_mb", "process_rss_mb",
    "battery_percent", "charging_state", "battery_saver", "thermal_status",
    "soc_temperature_c", "gpu_temperature_c", "battery_temperature_c",
    "screen_policy", "background_load_policy", "error",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def finite_float(value: str, label: str, line: int, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"row {line}: invalid {label} {value!r}") from error
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise ValueError(f"row {line}: non-finite {label}")
    if positive and parsed <= 0:
        raise ValueError(f"row {line}: {label} must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--expected-trials", default=3, type=int, choices=(2, 3))
    parser.add_argument("--app-manifest", required=True, type=Path)
    parser.add_argument("--flutter-root", type=Path, default=Path("flutter_app"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/final_benchmark_audit.json")
    )
    args = parser.parse_args()

    status = json.loads(args.status.read_text(encoding="utf-8"))
    expected_status = "COMPLETE" if args.expected_trials == 3 else "BOUNDED_EXPLORATORY_COMPLETE"
    if status.get("status") != expected_status:
        raise ValueError(f"device/derivation status is not {expected_status}")
    if Path(str(status.get("raw_csv", ""))).name != args.raw.name:
        raise ValueError("device status points to a different raw CSV basename")
    finished = parse_time(str(status["finished_utc"]))

    manifest = json.loads(args.app_manifest.read_text(encoding="utf-8"))
    frozen_values = {
        "status": "FROZEN_FINAL_DEVICE_BENCHMARK",
        "protocol_version": "1.2.0",
        "warmup_runs": 20,
        "measured_runs": 100,
        "complete_trials": 3,
        "random_seed": 42,
        "threads": [1, 2, 4],
        "runtimes": ["builtin_cpu", "xnnpack_cpu"],
    }
    for key, expected in frozen_values.items():
        if manifest.get(key) != expected:
            raise ValueError(f"app manifest {key!r} differs from frozen value")
    models = {str(model["id"]): model for model in manifest["models"]}
    if len(models) != 12:
        raise ValueError("app manifest must contain 12 distinct models")
    model_axes = {
        (str(model["architecture"]), str(model["quantization"]))
        for model in models.values()
    }
    if len(model_axes) != 12:
        raise ValueError("app manifest architecture/quantization matrix is incomplete")
    asset_hashes: dict[str, str] = {}
    for model_id, model in models.items():
        asset = args.flutter_root / str(model["asset"])
        if not asset.is_file():
            raise FileNotFoundError(asset)
        actual = sha256(asset)
        if actual != model["sha256"]:
            raise ValueError(f"app asset hash mismatch: {asset}")
        asset_hashes[model_id] = actual

    observations: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    errors: dict[tuple[str, str, int, int], list[str]] = defaultdict(list)
    order_indices: dict[tuple[str, str, int, int], set[int]] = defaultdict(set)
    devices: set[str] = set()
    charging_states: set[str] = set()
    temperatures: list[float] = []
    pss_values: list[float] = []
    rss_values: list[float] = []
    row_count = 0
    measured_count = 0
    error_count = 0
    first_timestamp: datetime | None = None
    previous_timestamp: datetime | None = None

    with args.raw.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SCHEMA:
            raise ValueError("raw CSV columns differ from the frozen protocol-1.2 schema")
        for line, row in enumerate(reader, start=2):
            row_count += 1
            timestamp = parse_time(row["timestamp_utc"])
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(f"row {line}: timestamps are not nondecreasing")
            first_timestamp = first_timestamp or timestamp
            previous_timestamp = timestamp
            if timestamp > finished:
                raise ValueError(f"row {line}: timestamp is after device completion time")
            if row["protocol_version"] != "1.2.0":
                raise ValueError(f"row {line}: unexpected protocol version")
            if row["app_version"] != "1.2.0+3" or row["build_mode"] != "release":
                raise ValueError(f"row {line}: unexpected app version/build mode")
            model_id = row["model_id"]
            if model_id not in models:
                raise ValueError(f"row {line}: unknown model id {model_id!r}")
            model = models[model_id]
            for field in (
                "model_sha256", "architecture", "quantization", "input_dtype",
                "output_dtype",
            ):
                manifest_field = "sha256" if field == "model_sha256" else field
                if row[field] != str(model[manifest_field]):
                    raise ValueError(f"row {line}: {field} differs from app manifest")
            if row["input_shape"] != "1x224x224x3":
                raise ValueError(f"row {line}: unexpected input shape")
            runtime = row["runtime"]
            if runtime not in manifest["runtimes"]:
                raise ValueError(f"row {line}: unknown runtime")
            requested = "xnnpack" if runtime == "xnnpack_cpu" else "none"
            effective = (
                "xnnpack_initialized_partition_unverified"
                if runtime == "xnnpack_cpu"
                else "none_builtin"
            )
            if row["requested_delegate"] != requested:
                raise ValueError(f"row {line}: requested delegate label mismatch")
            if row["effective_delegate"] != effective:
                raise ValueError(f"row {line}: effective delegate label mismatch")
            threads = int(row["threads"])
            trial = int(row["trial_id"])
            order_index = int(row["randomized_order_index"])
            if threads not in manifest["threads"] or not 1 <= trial <= args.expected_trials:
                raise ValueError(f"row {line}: thread/trial outside frozen matrix")
            key = (model_id, runtime, threads, trial)
            order_indices[key].add(order_index)
            devices.add(row["device_id"])
            charging_states.add(row["charging_state"])
            if row["device_id"] != "realme_RMX3760_api35":
                raise ValueError(f"row {line}: unexpected device identifier")
            if row["battery_saver"].lower() != "false":
                raise ValueError(f"row {line}: battery saver was not off")
            if row["screen_policy"] != "FLAG_KEEP_SCREEN_ON":
                raise ValueError(f"row {line}: screen policy differs from frozen protocol")
            if row["background_load_policy"] != "no_load_injected_background_apps_unverified":
                raise ValueError(f"row {line}: background-load label differs from protocol")
            battery = finite_float(row["battery_percent"], "battery percent", line)
            if not 0 <= battery <= 100:
                raise ValueError(f"row {line}: battery percent outside [0,100]")
            if not row["charging_state"]:
                raise ValueError(f"row {line}: charging state is missing")
            if int(row["thermal_status"]) < 0:
                raise ValueError(f"row {line}: invalid thermal status")
            temperatures.append(
                finite_float(row["battery_temperature_c"], "battery temperature", line)
            )
            pss_values.append(
                finite_float(row["process_pss_mb"], "process PSS", line, positive=True)
            )
            rss_values.append(
                finite_float(row["process_rss_mb"], "process RSS", line, positive=True)
            )
            finite_float(row["model_load_ms"], "model load time", line)

            phase = row["phase"]
            if phase == "measured":
                measured_count += 1
                run_index = int(row["run_index"])
                if not 0 <= run_index < 100:
                    raise ValueError(f"row {line}: measured run index outside [0,99]")
                finite_float(row["latency_ms"], "latency", line, positive=True)
                if row["error"] or row["delegate_error"]:
                    raise ValueError(f"row {line}: measured row contains an error")
                observations[key].append(run_index)
            elif phase == "configuration_error":
                error_count += 1
                if int(row["run_index"]) != -1 or row["latency_ms"]:
                    raise ValueError(f"row {line}: malformed configuration error row")
                if not row["error"]:
                    raise ValueError(f"row {line}: configuration error text is empty")
                errors[key].append(row["error"])
            else:
                raise ValueError(f"row {line}: unsupported phase {phase!r}")

    if row_count == 0 or first_timestamp is None or previous_timestamp is None:
        raise ValueError("raw benchmark contains no data rows")
    if len(devices) != 1:
        raise ValueError("raw benchmark contains more than one device")
    expected_keys = {
        (model_id, runtime, threads, trial)
        for model_id in models
        for runtime in manifest["runtimes"]
        for threads in manifest["threads"]
        for trial in range(1, args.expected_trials + 1)
    }
    if set(observations) | set(errors) != expected_keys:
        missing = sorted(expected_keys - (set(observations) | set(errors)))
        extra = sorted((set(observations) | set(errors)) - expected_keys)
        raise ValueError(f"raw matrix coverage mismatch; missing={missing}, extra={extra}")

    successful_configurations = 0
    failed_configurations = 0
    for model_id in models:
        for runtime in manifest["runtimes"]:
            for threads in manifest["threads"]:
                keys = [
                    (model_id, runtime, threads, trial)
                    for trial in range(1, args.expected_trials + 1)
                ]
                if all(
                    sorted(observations[key]) == list(range(100)) and not errors[key]
                    for key in keys
                ):
                    successful_configurations += 1
                elif all(not observations[key] and len(errors[key]) == 1 for key in keys):
                    failed_configurations += 1
                else:
                    raise ValueError(
                        f"mixed/incomplete configuration state: {model_id}/{runtime}/{threads}t"
                    )
                if any(len(order_indices[key]) != 1 for key in keys):
                    raise ValueError(
                        f"order index changed within configuration: {model_id}/{runtime}/{threads}t"
                    )
    for trial in range(1, args.expected_trials + 1):
        trial_orders = {
            next(iter(order_indices[key]))
            for key in expected_keys
            if key[3] == trial
        }
        if trial_orders != set(range(72)):
            raise ValueError(f"trial {trial}: randomized order is not a permutation of 0..71")

    expected_rows = (
        successful_configurations * args.expected_trials * 100
        + failed_configurations * args.expected_trials
    )
    if row_count != expected_rows:
        raise ValueError("raw row count differs from successful/error matrix accounting")
    report = {
        "status": "PASS",
        "raw_csv": str(args.raw),
        "raw_csv_sha256": sha256(args.raw),
        "device_status": str(args.status),
        "device_status_sha256": sha256(args.status),
        "app_manifest": str(args.app_manifest),
        "app_manifest_sha256": sha256(args.app_manifest),
        "device_id": next(iter(devices)),
        "first_timestamp_utc": first_timestamp.isoformat(),
        "last_timestamp_utc": previous_timestamp.isoformat(),
        "device_finished_utc": finished.isoformat(),
        "data_rows": row_count,
        "measured_rows": measured_count,
        "configuration_error_rows": error_count,
        "successful_configurations": successful_configurations,
        "failed_configurations": failed_configurations,
        "complete_trials": args.expected_trials,
        "planned_complete_trials": 3,
        "analysis_scope": (
            "confirmatory_complete_matrix"
            if args.expected_trials == 3
            else "post_hoc_bounded_exploratory_not_confirmatory"
        ),
        "measured_runs_per_successful_trial": 100,
        "charging_states": sorted(charging_states),
        "battery_temperature_min_c": min(temperatures),
        "battery_temperature_max_c": max(temperatures),
        "process_pss_min_mb": min(pss_values),
        "process_pss_max_mb": max(pss_values),
        "process_rss_min_mb": min(rss_values),
        "process_rss_max_mb": max(rss_values),
        "asset_hashes": asset_hashes,
        "test_split_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
