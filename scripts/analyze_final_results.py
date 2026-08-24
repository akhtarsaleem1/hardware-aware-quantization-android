#!/usr/bin/env python3
"""Validate final evidence and generate publication tables and figures.

The complete benchmark trial is the experimental unit. Per-inference rows are
retained for distributional summaries, but uncertainty is estimated by
resampling complete trial summaries so repeated invocations are not treated as
independent biological or device replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


QUANTIZATION_ORDER = ["float32", "float16", "dynamic_int8", "full_int8"]
RUNTIME_LABELS = {"builtin_cpu": "Built-in CPU", "xnnpack_cpu": "XNNPACK CPU"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variant_name(value: str) -> str:
    name = Path(value).stem.lower()
    for variant in ("dynamic_int8", "full_int8", "float16", "float32"):
        if variant in name:
            return variant
    raise ValueError(f"Cannot identify quantization variant from {value!r}")


def percentile95(values: pd.Series) -> float:
    return float(np.percentile(values.to_numpy(dtype=float), 95))


def bootstrap_trial_mean_ci(
    values: np.ndarray, rng: np.random.Generator, iterations: int = 20_000
) -> tuple[float, float]:
    if values.size < 2:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    estimates = values[indices].mean(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def load_accuracy(test_report_path: Path) -> pd.DataFrame:
    report = json.loads(test_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "FINAL_TEST_SINGLE_SESSION_COMPLETE":
        raise ValueError("The locked-test report is not marked complete")
    rows: list[dict[str, object]] = []
    for architecture, model in report["models"].items():
        for key, metrics in model["variants"].items():
            if key == "keras_reference":
                continue
            rows.append(
                {
                    "architecture": architecture,
                    "quantization": variant_name(str(metrics.get("path", key))),
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "top1_agreement_with_keras": metrics[
                        "top1_agreement_with_keras"
                    ],
                    "test_artifact_sha256": metrics["sha256"],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["architecture", "quantization"]).any():
        raise ValueError("Duplicate architecture/quantization test metrics")
    return frame


def load_selection_evidence(selection_manifest_path: Path) -> pd.DataFrame:
    manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_FOR_FINAL_TEST":
        raise ValueError("Selection manifest is not frozen for final test")
    rows: list[dict[str, object]] = []
    for model in manifest["models"]:
        architecture = model["architecture"]
        for quantization, metrics in model["validation_variants"].items():
            rows.append(
                {
                    "architecture": architecture,
                    "quantization": quantization,
                    "selection_validation_accuracy": metrics["validation_accuracy"],
                    "selection_validation_top1_agreement": metrics[
                        "top1_agreement_with_keras"
                    ],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["architecture", "quantization"]).any():
        raise ValueError("Duplicate validation selection evidence")
    return frame


def load_artifacts(manifest_path: Path, flutter_root: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for model in manifest["models"]:
        asset = flutter_root / model["asset"]
        if not asset.is_file():
            raise FileNotFoundError(asset)
        actual_hash = sha256(asset)
        if actual_hash != model["sha256"]:
            raise ValueError(f"Asset hash mismatch: {asset}")
        rows.append(
            {
                "model_id": model["id"],
                "architecture": model["architecture"],
                "quantization": model["quantization"],
                "model_sha256": actual_hash,
                "artifact_size_bytes": asset.stat().st_size,
                "artifact_size_mb": asset.stat().st_size / (1024 * 1024),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["architecture", "quantization"]).any():
        raise ValueError("Duplicate architecture/quantization artifacts")
    return frame


def trial_summaries(measured: pd.DataFrame) -> pd.DataFrame:
    keys = [
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
    rows: list[dict[str, object]] = []
    for key, group in measured.groupby(keys, dropna=False, sort=True):
        latency = group["latency_ms"].to_numpy(dtype=float)
        rows.append(
            dict(zip(keys, key))
            | {
                "observation_count": len(group),
                "latency_min_ms": latency.min(),
                "latency_max_ms": latency.max(),
                "latency_mean_ms": latency.mean(),
                "latency_median_ms": np.median(latency),
                "latency_p90_ms": np.percentile(latency, 90),
                "latency_p95_ms": np.percentile(latency, 95),
                "latency_p99_ms": np.percentile(latency, 99),
                "latency_sd_ms": latency.std(ddof=1),
                "latency_cv": latency.std(ddof=1) / latency.mean(),
                "throughput_images_per_s": 1000.0 / latency.mean(),
                "process_pss_median_mb": group["process_pss_mb"].median(),
                "process_rss_median_mb": group["process_rss_mb"].median(),
                "battery_temperature_start_c": group.iloc[0][
                    "battery_temperature_c"
                ],
                "battery_temperature_end_c": group.iloc[-1][
                    "battery_temperature_c"
                ],
                "battery_temperature_max_c": group[
                    "battery_temperature_c"
                ].max(),
            }
        )
    return pd.DataFrame(rows)


def configuration_summaries(
    trials: pd.DataFrame, measured: pd.DataFrame, seed: int
) -> pd.DataFrame:
    keys = [
        "device_id",
        "model_id",
        "model_sha256",
        "architecture",
        "quantization",
        "runtime",
        "requested_delegate",
        "effective_delegate",
        "threads",
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for key, trial_group in trials.groupby(keys, dropna=False, sort=True):
        raw_filter = np.ones(len(measured), dtype=bool)
        for column, value in zip(keys, key):
            raw_filter &= measured[column].to_numpy() == value
        raw = measured.loc[raw_filter, "latency_ms"].to_numpy(dtype=float)
        medians = trial_group["latency_median_ms"].to_numpy(dtype=float)
        p95s = trial_group["latency_p95_ms"].to_numpy(dtype=float)
        median_low, median_high = bootstrap_trial_mean_ci(medians, rng)
        p95_low, p95_high = bootstrap_trial_mean_ci(p95s, rng)
        rows.append(
            dict(zip(keys, key))
            | {
                "complete_trial_count": len(trial_group),
                "observation_count": len(raw),
                "latency_median_all_ms": np.median(raw),
                "latency_p95_all_ms": np.percentile(raw, 95),
                "trial_median_mean_ms": medians.mean(),
                "trial_median_sd_ms": medians.std(ddof=1),
                "trial_median_cv": medians.std(ddof=1) / medians.mean(),
                "trial_throughput_mean_images_per_s": trial_group[
                    "throughput_images_per_s"
                ].mean(),
                "trial_throughput_sd_images_per_s": trial_group[
                    "throughput_images_per_s"
                ].std(ddof=1),
                "trial_median_bootstrap95_low_ms": median_low,
                "trial_median_bootstrap95_high_ms": median_high,
                "trial_p95_mean_ms": p95s.mean(),
                "trial_p95_sd_ms": p95s.std(ddof=1),
                "trial_p95_bootstrap95_low_ms": p95_low,
                "trial_p95_bootstrap95_high_ms": p95_high,
                "process_pss_median_mb": trial_group[
                    "process_pss_median_mb"
                ].median(),
                "process_rss_median_mb": trial_group[
                    "process_rss_median_mb"
                ].median(),
                "battery_temperature_min_c": trial_group[
                    "battery_temperature_start_c"
                ].min(),
                "battery_temperature_max_c": trial_group[
                    "battery_temperature_max_c"
                ].max(),
            }
        )
    return pd.DataFrame(rows)


def pareto_mask(frame: pd.DataFrame, latency_column: str) -> np.ndarray:
    values = frame[
        ["selection_validation_accuracy", "artifact_size_bytes", latency_column]
    ].to_numpy(dtype=float)
    transformed = values.copy()
    transformed[:, 1:] *= -1
    efficient = np.ones(len(frame), dtype=bool)
    for candidate in range(len(frame)):
        at_least = np.all(transformed >= transformed[candidate], axis=1)
        strict = np.any(transformed > transformed[candidate], axis=1)
        efficient[candidate] = not np.any(at_least & strict)
    return efficient


def rank_stability(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["architecture", "runtime", "threads"]
    for key, group in trials.groupby(keys, sort=True):
        pivot = group.pivot(
            index="quantization", columns="trial_id", values="latency_median_ms"
        ).reindex(QUANTIZATION_ORDER)
        pivot = pivot.dropna(axis=0, how="all")
        for first, second in combinations(pivot.columns, 2):
            usable = pivot[[first, second]].dropna()
            coefficient = (
                spearmanr(usable[first], usable[second]).statistic
                if len(usable) >= 2
                else np.nan
            )
            rows.append(
                dict(zip(keys, key))
                | {
                    "trial_a": first,
                    "trial_b": second,
                    "configuration_count": len(usable),
                    "spearman_rank_correlation": coefficient,
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figures(results: pd.DataFrame, rank: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    output.mkdir(parents=True, exist_ok=True)

    model_level = results.drop_duplicates(["architecture", "quantization"])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.scatterplot(
        data=model_level,
        x="artifact_size_mb",
        y="macro_f1",
        hue="architecture",
        style="quantization",
        s=95,
        ax=ax,
    )
    ax.set(xlabel="TFLite artifact size (MiB)", ylabel="Locked-test macro F1")
    save_figure(fig, output / "figure_1_accuracy_size")

    architecture_labels = {
        "efficientnet_b0": "EfficientNet-B0",
        "mobilenet_v2": "MobileNetV2",
        "mobilenet_v3_small": "MobileNetV3-Small",
    }
    plot_results = results.assign(
        architecture_label=results["architecture"].map(architecture_labels),
        thread_label=results["threads"].map(
            lambda value: f"{value} thread" if value == 1 else f"{value} threads"
        ),
    )
    for number, metric, label in (
        (2, "trial_median_mean_ms", "Mean of trial medians (ms)"),
        (3, "trial_p95_mean_ms", "Mean of trial p95 latency (ms)"),
    ):
        grid = sns.catplot(
            data=plot_results,
            x="quantization",
            y=metric,
            hue="runtime",
            col="architecture_label",
            row="thread_label",
            kind="bar",
            sharey=False,
            height=1.75,
            aspect=1.45,
        )
        grid.set_axis_labels("", "")
        grid.set_titles(template="{col_name} | {row_name}", size=9)
        grid.figure.set_size_inches(10.2, 5.9)
        grid.figure.supylabel(label, x=0.015, fontsize=10)
        for axis in grid.axes.flat:
            axis.tick_params(axis="x", labelrotation=22, labelsize=7)
            axis.tick_params(axis="y", labelsize=7)
        grid.figure.subplots_adjust(
            top=0.92, bottom=0.18, left=0.08, right=0.88,
            hspace=0.55, wspace=0.28,
        )
        save_figure(grid.figure, output / f"figure_{number}_latency")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.scatterplot(
        data=results,
        x="trial_median_mean_ms",
        y="selection_validation_accuracy",
        hue="architecture",
        style="quantization",
        size="threads",
        sizes=(45, 130),
        alpha=0.8,
        ax=ax,
    )
    front = results[results["pareto_median"]]
    ax.scatter(
        front["trial_median_mean_ms"],
        front["selection_validation_accuracy"],
        facecolors="none",
        edgecolors="black",
        s=180,
        linewidths=1.2,
        label="Median-based Pareto front",
    )
    ax.set(
        xlabel="Mean of trial medians (ms)",
        ylabel="Frozen-validation accuracy used for selection",
    )
    save_figure(fig, output / "figure_4_pareto")

    if not rank.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        order = rank.assign(
            context=lambda value: value["architecture"]
            + " · "
            + value["runtime"]
            + " · "
            + value["threads"].astype(str)
            + "t"
        )
        sns.stripplot(
            data=order,
            x="spearman_rank_correlation",
            y="context",
            orient="h",
            jitter=False,
            size=5,
            color="#2E74B5",
            ax=ax,
        )
        ax.set(xlim=(-1.05, 1.05), xlabel="Pairwise Spearman rank correlation", ylabel="")
        save_figure(fig, output / "figure_5_rank_stability")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--benchmark-audit", required=True, type=Path)
    parser.add_argument("--test-report", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--app-manifest", required=True, type=Path)
    parser.add_argument("--flutter-root", default=Path("flutter_app"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/final"), type=Path)
    parser.add_argument("--expected-runs", default=100, type=int)
    parser.add_argument("--expected-trials", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    benchmark_audit = json.loads(args.benchmark_audit.read_text(encoding="utf-8"))
    if benchmark_audit.get("status") != "PASS":
        raise ValueError("Independent benchmark audit is not PASS")
    if benchmark_audit.get("raw_csv_sha256") != sha256(args.raw):
        raise ValueError("Benchmark audit does not hash-link the analyzed raw CSV")
    if benchmark_audit.get("test_split_accessed") is not False:
        raise ValueError("Benchmark audit does not attest that the test split stayed locked")

    raw = pd.read_csv(args.raw)
    if list(raw.columns) != [
        "timestamp_utc", "protocol_version", "device_id", "app_version",
        "build_mode", "model_id", "model_sha256", "architecture",
        "quantization", "input_dtype", "output_dtype", "input_shape",
        "runtime", "requested_delegate", "effective_delegate", "delegate_error",
        "threads", "trial_id", "randomized_order_index", "phase", "run_index",
        "latency_ms", "model_load_ms", "process_pss_mb", "process_rss_mb",
        "battery_percent", "charging_state", "battery_saver", "thermal_status",
        "soc_temperature_c", "gpu_temperature_c", "battery_temperature_c",
        "screen_policy", "background_load_policy", "error",
    ]:
        raise ValueError("Raw benchmark columns differ from the frozen schema")
    measured = raw[raw["phase"] == "measured"].copy()
    errors = raw[raw["phase"] == "configuration_error"].copy()
    if len(measured) == 0:
        raise ValueError("No measured benchmark observations")
    for column in (
        "latency_ms", "process_pss_mb", "process_rss_mb", "battery_temperature_c"
    ):
        measured[column] = pd.to_numeric(measured[column], errors="raise")
    if (measured["latency_ms"] <= 0).any():
        raise ValueError("Measured latency must be positive")

    count_keys = ["model_id", "runtime", "threads", "trial_id"]
    counts = measured.groupby(count_keys).size()
    if not (counts == args.expected_runs).all():
        raise ValueError(
            "Incomplete measured configuration/trial blocks: "
            + counts[counts != args.expected_runs].to_string()
        )
    successful_trials = measured.groupby(["model_id", "runtime", "threads"])[
        "trial_id"
    ].nunique()
    if not (successful_trials == args.expected_trials).all():
        raise ValueError("At least one successful configuration lacks complete trials")

    trials = trial_summaries(measured)
    configurations = configuration_summaries(trials, measured, args.seed)
    accuracy = load_accuracy(args.test_report)
    selection = load_selection_evidence(args.selection_manifest)
    artifacts = load_artifacts(args.app_manifest, args.flutter_root)
    results = configurations.merge(
        artifacts,
        on=["model_id", "architecture", "quantization", "model_sha256"],
        validate="many_to_one",
    ).merge(
        selection,
        on=["architecture", "quantization"],
        validate="many_to_one",
    ).merge(
        accuracy,
        on=["architecture", "quantization"],
        validate="many_to_one",
    )
    if len(results) != len(configurations):
        raise ValueError("Not all benchmark configurations matched artifacts and accuracy")
    results["configuration_id"] = (
        results["model_id"]
        + "__"
        + results["runtime"]
        + "__"
        + results["threads"].astype(str)
        + "t"
    )
    results["pareto_median"] = pareto_mask(results, "trial_median_mean_ms")
    results["pareto_p95"] = pareto_mask(results, "trial_p95_mean_ms")
    rank = rank_stability(trials)

    model_results = results.sort_values(
        ["architecture", "quantization", "runtime", "threads"]
    ).drop_duplicates(["architecture", "quantization"])[
        [
            "architecture", "quantization", "model_sha256",
            "artifact_size_bytes", "artifact_size_mb",
            "selection_validation_accuracy", "selection_validation_top1_agreement",
            "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "top1_agreement_with_keras", "test_artifact_sha256",
        ]
    ].copy()
    fp32_sizes = model_results[model_results["quantization"] == "float32"].set_index(
        "architecture"
    )["artifact_size_bytes"].to_dict()
    model_results["compression_percent_vs_float32"] = model_results.apply(
        lambda row: 100.0
        * (1.0 - row["artifact_size_bytes"] / fp32_sizes[row["architecture"]]),
        axis=1,
    )
    for front_name in ("pareto_median", "pareto_p95"):
        front_any = results.groupby(["architecture", "quantization"])[front_name].any()
        model_results[f"any_configuration_{front_name}"] = [
            bool(front_any.loc[(row.architecture, row.quantization)])
            for row in model_results.itertuples()
        ]

    size_latency_contexts = 0
    size_latency_rank_mismatches = 0
    for _, group in results.groupby(["architecture", "runtime", "threads"]):
        size_order = tuple(group.sort_values("artifact_size_bytes")["quantization"])
        latency_order = tuple(group.sort_values("trial_median_mean_ms")["quantization"])
        size_latency_contexts += 1
        size_latency_rank_mismatches += int(size_order != latency_order)
    runtime_rank_contexts = 0
    runtime_rank_changes = 0
    for _, group in results.groupby(["architecture", "threads"]):
        orders = [
            tuple(runtime_group.sort_values("trial_median_mean_ms")["quantization"])
            for _, runtime_group in group.groupby("runtime")
        ]
        if len(orders) == 2:
            runtime_rank_contexts += 1
            runtime_rank_changes += int(orders[0] != orders[1])
    runtime_cv = results.groupby("runtime")["trial_median_cv"].median()
    cv_ratio = float(runtime_cv.max() / runtime_cv.min()) if runtime_cv.min() > 0 else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_dir / "table_trial_summaries.csv", index=False)
    results.to_csv(args.output_dir / "table_configuration_results.csv", index=False)
    model_results.to_csv(args.output_dir / "table_model_results.csv", index=False)
    errors.to_csv(args.output_dir / "table_configuration_errors.csv", index=False)
    rank.to_csv(args.output_dir / "table_rank_stability.csv", index=False)
    figures(results, rank, args.output_dir / "figures")

    median_front = results[results["pareto_median"]]
    p95_front = results[results["pareto_p95"]]
    summary = {
        "status": "FINAL_ANALYSIS_COMPLETE",
        "raw_csv": str(args.raw),
        "raw_csv_sha256": sha256(args.raw),
        "benchmark_audit": str(args.benchmark_audit),
        "benchmark_audit_sha256": sha256(args.benchmark_audit),
        "test_report": str(args.test_report),
        "test_report_sha256": sha256(args.test_report),
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "pareto_selection_uses_validation_not_test": True,
        "measured_observations": len(measured),
        "configuration_error_rows": len(errors),
        "successful_configurations": len(results),
        "complete_trials_per_successful_configuration": args.expected_trials,
        "independent_device_count": int(measured["device_id"].nunique()),
        "inference_rows_are_not_treated_as_independent_device_replicates": True,
        "median_pareto_configuration_ids": median_front["configuration_id"].tolist(),
        "p95_pareto_configuration_ids": p95_front["configuration_id"].tolist(),
        "rank_stability_spearman_min": float(
            rank["spearman_rank_correlation"].min()
        ) if not rank.empty else None,
        "rank_stability_spearman_median": float(
            rank["spearman_rank_correlation"].median()
        ) if not rank.empty else None,
        "hypothesis_evidence": {
            "H1_size_vs_latency_rank_mismatch_contexts": size_latency_rank_mismatches,
            "H1_total_contexts": size_latency_contexts,
            "H2_runtime_rank_change_contexts": runtime_rank_changes,
            "H2_total_comparable_contexts": runtime_rank_contexts,
            "H3_dominated_int8_configurations_median_front": int(
                ((results["quantization"].isin(["dynamic_int8", "full_int8"]))
                 & ~results["pareto_median"]).sum()
            ),
            "H4_median_trial_cv_by_runtime": {
                str(key): float(value) for key, value in runtime_cv.items()
            },
            "H4_max_to_min_runtime_cv_ratio": cv_ratio,
            "interpretation_policy": "descriptive_complete_trial_evidence_not_automatic_significance_claims",
        },
        "limitations": [
            "one physical Android device; no hardware-level generalization",
            f"{args.expected_trials} complete trials yield imprecise trial-level uncertainty",
            "USB charging and background applications were recorded but not fully controlled",
            "process PSS/RSS are observational snapshots, not isolated model peak memory",
        ],
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
