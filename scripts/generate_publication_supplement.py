#!/usr/bin/env python3
"""Generate the nine requested evidence tables and supplemental journal figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def write_table(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise ValueError(f"refusing to write empty publication table: {path.name}")
    frame.to_csv(path, index=False)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--results-dir", type=Path, default=Path("results/final"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final/publication_supplement"))
    args = parser.parse_args()
    root = args.root.resolve()
    results = resolve(root, str(args.results_dir))
    output = resolve(root, str(args.output_dir))
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "FINAL_ANALYSIS_COMPLETE":
        raise ValueError("final analysis is not complete")
    raw_path = resolve(root, str(summary["raw_csv"]))
    if sha256(raw_path) != summary["raw_csv_sha256"]:
        raise ValueError("raw benchmark hash differs from final analysis")
    audit_path = resolve(root, str(summary["benchmark_audit"]))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("raw_csv_sha256") != sha256(raw_path):
        raise ValueError("benchmark audit is missing or not hash-linked")

    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    dataset = json.loads((root / "reports/dataset_report.json").read_text(encoding="utf-8"))
    if dataset.get("status") != "PASS":
        raise ValueError("grouped dataset report is not PASS")
    models = pd.read_csv(results / "table_model_results.csv")
    configurations = pd.read_csv(results / "table_configuration_results.csv")
    trials = pd.read_csv(results / "table_trial_summaries.csv")
    errors = pd.read_csv(results / "table_configuration_errors.csv")
    raw = pd.read_csv(raw_path)
    measured = raw[raw["phase"] == "measured"].copy()
    if len(measured) != int(summary["measured_observations"]):
        raise ValueError("measured raw row count differs from final analysis")

    training_rows = [
        ("dataset", "name", config["dataset"]["name"]),
        ("dataset", "license", config["dataset"]["license"]),
        ("dataset", "records", dataset["record_count"]),
        ("dataset", "train_images", dataset["split_counts"]["train"]),
        ("dataset", "validation_images", dataset["split_counts"]["validation"]),
        ("dataset", "test_images", dataset["split_counts"]["test"]),
        ("dataset", "split_protocol", config["dataset"]["split_protocol"]),
        ("dataset", "input_size", "x".join(map(str, config["dataset"]["image_size"]))),
        ("training", "random_seed", config["project"]["random_seed"]),
        ("training", "framework", config["training"]["framework"]),
        ("training", "batch_size", config["training"]["batch_size"]),
        ("training", "maximum_epochs", config["training"]["epochs"]),
        ("training", "frozen_backbone_epochs", config["training"]["frozen_backbone_epochs"]),
        ("training", "optimizer", config["training"]["optimizer"]),
        ("training", "initial_learning_rate", config["training"]["learning_rate"]),
        ("training", "fine_tuning_learning_rate", config["training"]["fine_tuning_learning_rate"]),
        ("training", "class_weighting", config["training"]["class_weighting"]),
        ("quantization", "variants", ";".join(config["quantization"]["variants"])),
        ("quantization", "representative_source", config["quantization"]["representative_source"]),
        ("benchmark", "matrix", "12 models x 2 runtimes x 3 thread counts"),
        ("benchmark", "trials_warmups_measured", "3 x (20 warmups + 100 measured)"),
    ]
    table1 = pd.DataFrame(training_rows, columns=["category", "item", "value"])

    metadata_rows = []
    for path in sorted((root / "models/metadata").glob("*/*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        model = str(value.get("model", ""))
        if model not in set(models["architecture"]):
            continue
        validation = models[models["architecture"] == model]["selection_validation_accuracy"]
        metadata_rows.append({
            "architecture": model,
            "run_id": value["run_id"],
            "parameter_count": value["parameter_count"],
            "pretraining": "ImageNet",
            "input_shape": "224x224x3",
            "best_frozen_validation_accuracy": validation.max(),
            "gpu": value["gpu_report"]["physical_gpus"][0]["device_name"],
            "model_sha256": models.loc[models["architecture"] == model, "model_sha256"].iloc[0],
        })
    table2 = pd.DataFrame(metadata_rows).drop_duplicates("architecture")
    if len(table2) != 3:
        raise ValueError(f"expected three architecture metadata rows, found {len(table2)}")

    table3 = models[[
        "architecture", "quantization", "selection_validation_accuracy",
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "top1_agreement_with_keras", "test_artifact_sha256",
    ]].copy()
    table4 = models[[
        "architecture", "quantization", "artifact_size_bytes", "artifact_size_mb",
        "compression_percent_vs_float32", "model_sha256",
    ]].copy()
    table5 = configurations[[
        "configuration_id", "architecture", "quantization", "runtime", "threads",
        "trial_median_mean_ms", "trial_median_sd_ms", "trial_median_cv",
        "trial_median_bootstrap95_low_ms", "trial_median_bootstrap95_high_ms",
        "trial_p95_mean_ms", "trial_p95_sd_ms", "trial_p95_bootstrap95_low_ms",
        "trial_p95_bootstrap95_high_ms", "trial_throughput_mean_images_per_s",
        "trial_throughput_sd_images_per_s", "observation_count",
    ]].copy()
    table6 = configurations[[
        "configuration_id", "architecture", "quantization", "runtime", "threads",
        "process_pss_median_mb", "process_rss_median_mb",
        "battery_temperature_min_c", "battery_temperature_max_c",
    ]].copy()
    table7 = configurations.groupby(["runtime", "threads"], as_index=False).agg(
        successful_configurations=("configuration_id", "count"),
        median_of_trial_medians_ms=("trial_median_mean_ms", "median"),
        median_of_trial_p95_ms=("trial_p95_mean_ms", "median"),
        median_trial_cv=("trial_median_cv", "median"),
        median_pss_mb=("process_pss_median_mb", "median"),
        median_rss_mb=("process_rss_median_mb", "median"),
    )
    table7["effective_delegate"] = table7["runtime"].map(
        {"builtin_cpu": "none_builtin", "xnnpack_cpu": "xnnpack_initialized_partition_unverified"}
    )
    table7["configuration_error_rows"] = table7.apply(
        lambda row: len(errors[(errors["runtime"] == row["runtime"]) & (errors["threads"] == row["threads"])]),
        axis=1,
    )
    table8 = configurations[
        configurations["pareto_median"] | configurations["pareto_p95"]
    ][[
        "configuration_id", "architecture", "quantization", "runtime", "threads",
        "selection_validation_accuracy", "artifact_size_mb", "trial_median_mean_ms",
        "trial_p95_mean_ms", "pareto_median", "pareto_p95",
    ]].copy()
    table9 = table8.copy()
    table9["recommendation_scope"] = "tested realme RMX3760 / protocol 1.2.0 only"
    table9["selection_quality_source"] = "frozen validation accuracy; locked test excluded"
    table9["recommendation_tier"] = table9.apply(
        lambda row: "median_and_p95" if row["pareto_median"] and row["pareto_p95"]
        else ("median_only" if row["pareto_median"] else "p95_only"), axis=1,
    )

    tables = {
        "table_1_dataset_training_configuration.csv": table1,
        "table_2_model_architecture_comparison.csv": table2,
        "table_3_accuracy_metrics.csv": table3,
        "table_4_quantization_model_sizes.csv": table4,
        "table_5_android_latency_statistics.csv": table5,
        "table_6_memory_measurements.csv": table6,
        "table_7_runtime_delegate_comparison.csv": table7,
        "table_8_pareto_efficient_configurations.csv": table8,
        "table_9_hardware_aware_recommendations.csv": table9,
    }
    for name, frame in tables.items():
        write_table(frame, tables_dir / name)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    architecture = table2.sort_values("parameter_count")
    sns.barplot(data=architecture, x="architecture", y="parameter_count", color="#2E74B5", ax=ax)
    ax.set(xlabel="Architecture", ylabel="Model parameter count")
    ax.tick_params(axis="x", rotation=15)
    save_figure(fig, figures_dir / "model_architecture")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.barplot(data=models, x="quantization", y="artifact_size_mb", hue="architecture", ax=ax)
    ax.set(xlabel="Quantization", ylabel="TFLite artifact size (MiB)")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, figures_dir / "model_size_comparison")

    grid = sns.catplot(
        data=measured, x="quantization", y="latency_ms", hue="runtime",
        col="architecture", row="threads", kind="box", sharey=False,
        showfliers=True, fliersize=0.7, height=2.1, aspect=1.2,
    )
    grid.set_axis_labels("", "Latency (ms)")
    grid.set_xticklabels(rotation=25, horizontalalignment="right")
    save_figure(grid.figure, figures_dir / "benchmark_distribution")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.boxplot(
        data=configurations, x="threads", y="trial_median_mean_ms", hue="runtime",
        showfliers=True, ax=ax,
    )
    ax.set(xlabel="Threads", ylabel="Mean complete-trial median latency (ms)")
    save_figure(fig, figures_dir / "runtime_comparison")

    aliases = {
        "latency_comparison": results / "figures/figure_2_latency",
        "accuracy_latency_tradeoff": results / "figures/figure_4_pareto",
        "pareto_front": results / "figures/figure_4_pareto",
    }
    for name, source_stem in aliases.items():
        for suffix in (".png", ".pdf"):
            source = source_stem.with_suffix(suffix)
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, (figures_dir / name).with_suffix(suffix))

    artifacts = sorted([*tables_dir.glob("*.csv"), *figures_dir.glob("*.png"), *figures_dir.glob("*.pdf")])
    manifest = {
        "status": "PUBLICATION_SUPPLEMENT_COMPLETE",
        "raw_benchmark": str(raw_path.relative_to(root)),
        "raw_benchmark_sha256": sha256(raw_path),
        "analysis_summary": str(summary_path.relative_to(root)),
        "analysis_summary_sha256": sha256(summary_path),
        "table_count": len(list(tables_dir.glob("*.csv"))),
        "figure_png_count": len(list(figures_dir.glob("*.png"))),
        "figure_pdf_count": len(list(figures_dir.glob("*.pdf"))),
        "artifacts": [
            {"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        ],
    }
    manifest_path = output / "supplement_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
