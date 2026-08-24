#!/usr/bin/env python3
"""Populate the journal manuscript from completed, hash-linked result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def ms(value: float) -> str:
    return f"{float(value):.2f}"


def mib(value: float) -> str:
    return f"{float(value):.2f}"


def safe(value: object) -> str:
    return str(value).replace("|", "/").replace("\n", " ").strip()


def markdown_table(columns: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(safe(value) for value in row) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def replace_section(text: str, heading: str, next_heading: str, body: str) -> str:
    pattern = re.compile(
        rf"(?ms)^{re.escape(heading)}\s*$.*?(?=^{re.escape(next_heading)}\s*$)"
    )
    replacement = f"{heading}\n\n{body.rstrip()}\n\n"
    value, count = pattern.subn(lambda _: replacement, text, count=1)
    if count != 1:
        raise ValueError(f"Could not replace manuscript section {heading!r}")
    return value


def front_text(values: set[str]) -> str:
    return ", ".join(sorted(values)) if values else "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=Path("paper/manuscript_main.md"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/final"))
    parser.add_argument("--benchmark-audit", type=Path, default=Path("reports/final_benchmark_audit.json"))
    parser.add_argument("--app-manifest", type=Path, default=Path("flutter_app/assets/benchmark_manifest.json"))
    parser.add_argument("--test-report", type=Path, default=Path("results/final/final_test_report.json"))
    args = parser.parse_args()

    summary = json.loads((args.results_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    audit = json.loads(args.benchmark_audit.read_text(encoding="utf-8"))
    test_report = json.loads(args.test_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.app_manifest.read_text(encoding="utf-8"))
    required_status = {
        "analysis": (summary.get("status"), "FINAL_ANALYSIS_COMPLETE"),
        "audit": (audit.get("status"), "PASS"),
        "test": (test_report.get("status"), "FINAL_TEST_SINGLE_SESSION_COMPLETE"),
    }
    for label, (actual, expected) in required_status.items():
        if actual != expected:
            raise ValueError(f"{label} status {actual!r}, expected {expected!r}")
    if summary.get("pareto_selection_uses_validation_not_test") is not True:
        raise ValueError("Manuscript finalization requires validation-only Pareto selection")
    if audit.get("test_split_accessed") is not False:
        raise ValueError("Benchmark audit does not attest a locked test split")

    models = pd.read_csv(args.results_dir / "table_model_results.csv")
    configs = pd.read_csv(args.results_dir / "table_configuration_results.csv")
    errors = pd.read_csv(args.results_dir / "table_configuration_errors.csv")
    rank = pd.read_csv(args.results_dir / "table_rank_stability.csv")
    if len(models) != 12:
        raise ValueError(f"Expected 12 model rows, found {len(models)}")
    if len(configs) != int(summary["successful_configurations"]):
        raise ValueError("Configuration table count differs from analysis summary")

    best_accuracy = models.loc[models["accuracy"].idxmax()]
    worst_accuracy = models.loc[models["accuracy"].idxmin()]
    best_f1 = models.loc[models["macro_f1"].idxmax()]
    fastest_median = configs.loc[configs["trial_median_mean_ms"].idxmin()]
    fastest_p95 = configs.loc[configs["trial_p95_mean_ms"].idxmin()]
    median_front = set(summary["median_pareto_configuration_ids"])
    p95_front = set(summary["p95_pareto_configuration_ids"])
    front_overlap = median_front & p95_front
    only_median = median_front - p95_front
    only_p95 = p95_front - median_front
    hypotheses = summary["hypothesis_evidence"]

    model_rows: list[list[object]] = []
    for row in models.sort_values(["architecture", "quantization"]).itertuples():
        model_rows.append([
            row.architecture,
            row.quantization,
            f"{int(row.artifact_size_bytes):,}",
            mib(row.artifact_size_mb),
            f"{float(row.compression_percent_vs_float32):.1f}%",
            pct(row.selection_validation_accuracy),
            pct(row.selection_validation_top1_agreement),
            pct(row.accuracy),
            pct(row.macro_precision),
            pct(row.macro_recall),
            pct(row.macro_f1),
            str(row.model_sha256)[:12],
        ])
    table1 = markdown_table(
        [
            "Architecture", "Variant", "Bytes", "MiB", "Compression vs FP32",
            "Val. acc.", "Val. agreement", "Test acc.", "Macro P", "Macro R",
            "Macro F1", "SHA-256 prefix",
        ],
        model_rows,
    )

    error_groups: dict[tuple[str, str, int], tuple[int, str]] = {}
    if not errors.empty:
        for key, group in errors.groupby(["model_id", "runtime", "threads"], sort=True):
            messages = sorted({safe(value) for value in group["error"] if safe(value)})
            error_groups[(str(key[0]), str(key[1]), int(key[2]))] = (
                int(group["trial_id"].nunique()),
                "; ".join(messages),
            )
    config_lookup = {
        (str(row.model_id), str(row.runtime), int(row.threads)): row
        for row in configs.itertuples()
    }
    config_rows: list[list[object]] = []
    for model in sorted(manifest["models"], key=lambda value: value["id"]):
        for runtime in manifest["runtimes"]:
            for threads in manifest["threads"]:
                key = (model["id"], runtime, int(threads))
                row = config_lookup.get(key)
                if row is not None:
                    config_rows.append([
                        model["id"], runtime, threads,
                        f"{ms(row.trial_median_mean_ms)} [{ms(row.trial_median_bootstrap95_low_ms)}, {ms(row.trial_median_bootstrap95_high_ms)}]",
                        f"{ms(row.trial_p95_mean_ms)} [{ms(row.trial_p95_bootstrap95_low_ms)}, {ms(row.trial_p95_bootstrap95_high_ms)}]",
                        f"{float(row.trial_median_cv):.4f}",
                        mib(row.process_pss_median_mb),
                        mib(row.process_rss_median_mb),
                        "success",
                    ])
                else:
                    trial_count, message = error_groups.get(key, (0, "missing error provenance"))
                    config_rows.append([
                        model["id"], runtime, threads, "NA", "NA", "NA", "NA", "NA",
                        f"failed {trial_count}/3 trials: {message}",
                    ])
    if len(config_rows) != 72:
        raise ValueError("Table 2 does not account for all 72 configurations")
    summary_rows: list[list[object]] = []
    for runtime in manifest["runtimes"]:
        for threads in manifest["threads"]:
            group = configs[
                (configs["runtime"] == runtime) & (configs["threads"] == int(threads))
            ]
            failed_count = sum(
                1
                for (_, failed_runtime, failed_threads) in error_groups
                if failed_runtime == runtime and failed_threads == int(threads)
            )
            summary_rows.append([
                runtime,
                threads,
                len(group),
                failed_count,
                f"{ms(group['trial_median_mean_ms'].median())} [{ms(group['trial_median_mean_ms'].min())}, {ms(group['trial_median_mean_ms'].max())}]",
                f"{ms(group['trial_p95_mean_ms'].median())} [{ms(group['trial_p95_mean_ms'].min())}, {ms(group['trial_p95_mean_ms'].max())}]",
                mib(group["process_pss_median_mb"].median()),
                mib(group["process_rss_median_mb"].median()),
            ])
    table2 = markdown_table(
        [
            "Runtime", "Threads", "Successful", "Failed",
            "Median of trial medians [range] ms",
            "Median of trial p95 [range] ms", "Median PSS MiB", "Median RSS MiB",
        ],
        summary_rows,
    )

    failed_descriptions = [
        f"{model_id}/{runtime}/{threads}t ({trials}/3 trials: {message})"
        for (model_id, runtime, threads), (trials, message) in sorted(error_groups.items())
    ]
    failed_text = "; ".join(failed_descriptions) if failed_descriptions else "none"

    classification_body = f"""All 12 frozen flatbuffers completed the locked 3,501-image evaluation. Test accuracy ranged from {pct(worst_accuracy['accuracy'])} ({worst_accuracy['architecture']} {worst_accuracy['quantization']}) to {pct(best_accuracy['accuracy'])} ({best_accuracy['architecture']} {best_accuracy['quantization']}); the highest macro F1 was {pct(best_f1['macro_f1'])} for {best_f1['architecture']} {best_f1['quantization']}. Artifact size ranged from {mib(models['artifact_size_mb'].min())} to {mib(models['artifact_size_mb'].max())} MiB. The low MobileNetV3-Small full-INT8 validation parity was retained before test opening and remained visible in the locked-test table rather than being removed post hoc.

**Table 1. Classification quality, size, validation gating, and artifact identity for every frozen TFLite variant.**

{table1}

Keras references and complete per-class metrics/confusion matrices remain in the hash-linked final-test JSON; Table 1 reports the deployable flatbuffers used on the phone.

![Figure 1. Locked-test macro F1 against TFLite artifact size for all 12 frozen variants; color identifies architecture and marker identifies quantization.](../results/final/figures/figure_1_accuracy_size.png)"""

    android_body = f"""The device run contained {int(audit['measured_rows']):,} timed observations from {int(audit['successful_configurations'])} successful configurations and {int(audit['configuration_error_rows'])} explicit error rows representing {int(audit['failed_configurations'])} failed configurations. The fastest mean trial-median latency was {ms(fastest_median['trial_median_mean_ms'])} ms for {fastest_median['configuration_id']}; the fastest mean trial-p95 latency was {ms(fastest_p95['trial_p95_mean_ms'])} ms for {fastest_p95['configuration_id']}. Across successful configurations, median process PSS snapshots ranged from {mib(configs['process_pss_median_mb'].min())} to {mib(configs['process_pss_median_mb'].max())} MiB and RSS snapshots ranged from {mib(configs['process_rss_median_mb'].min())} to {mib(configs['process_rss_median_mb'].max())} MiB.

The preserved failed configurations were: {failed_text}. Battery temperature ranged from {float(audit['battery_temperature_min_c']):.1f} to {float(audit['battery_temperature_max_c']):.1f} °C, with charging states {', '.join(audit['charging_states'])}. Because USB power remained connected and the preferred ≤35 °C start condition was not met, these observations are not described as a cold or thermally controlled run.

**Table 2. Runtime/thread summary across successful model-quantization configurations. Brackets give the observed configuration range; the complete 72-configuration ledger is Supplementary Table 5.**

{table2}

The complete 72-configuration ledger, trial-level intervals, throughput, and error rows are provided in the hash-linked supplement. Figures 2 and 3 show the successful configurations for median and p95 latency without replacing unsupported configurations with fallback values.

![Figure 2. Mean complete-trial median latency by architecture, quantization, runtime, and thread count for every successful configuration.](../results/final/figures/figure_2_latency.png)

![Figure 3. Mean complete-trial p95 latency by architecture, quantization, runtime, and thread count for every successful configuration.](../results/final/figures/figure_3_latency.png)"""

    rank_body = f"""Artifact-size and latency orders differed in {int(hypotheses['H1_size_vs_latency_rank_mismatch_contexts'])} of {int(hypotheses['H1_total_contexts'])} architecture/runtime/thread contexts. Quantization order changed between built-in and XNNPACK paths in {int(hypotheses['H2_runtime_rank_change_contexts'])} of {int(hypotheses['H2_total_comparable_contexts'])} comparable architecture/thread contexts. Pairwise complete-trial Spearman correlations ranged from {float(summary['rank_stability_spearman_min']):.3f} to 1.000, with median {float(summary['rank_stability_spearman_median']):.3f}.

The validation-quality/size/latency Pareto analysis identified {len(median_front)} median-based and {len(p95_front)} p95-based configurations, with {len(front_overlap)} shared. Median-only configurations were {front_text(only_median)}; p95-only configurations were {front_text(only_p95)}. Thus the selected frontier changed when tail latency replaced median latency. Figures 4 and 5 visualize the validation-only Pareto surface and complete-trial rank stability; locked-test metrics are reporting outcomes and were not selection inputs.

![Figure 4. Frozen-validation accuracy versus mean complete-trial median latency; outlined points are on the three-objective validation/size/latency Pareto front.](../results/final/figures/figure_4_pareto.png)

![Figure 5. Pairwise Spearman stability of quantization latency ranks across complete trials, stratified by architecture, runtime, and thread count.](../results/final/figures/figure_5_rank_stability.png)"""

    h1 = "supported descriptively" if hypotheses["H1_size_vs_latency_rank_mismatch_contexts"] > 0 else "not supported"
    h2 = "supported descriptively" if hypotheses["H2_runtime_rank_change_contexts"] > 0 else "not supported"
    h3 = "supported descriptively" if hypotheses["H3_dominated_int8_configurations_median_front"] > 0 else "not supported"
    cv_values = hypotheses["H4_median_trial_cv_by_runtime"]
    discussion_body = f"""The results separate artifact compression from realized execution. H1 was {h1}: size order and median-latency order disagreed in {int(hypotheses['H1_size_vs_latency_rank_mismatch_contexts'])}/{int(hypotheses['H1_total_contexts'])} contexts. H2 was {h2}: runtime changed the quantization ordering in {int(hypotheses['H2_runtime_rank_change_contexts'])}/{int(hypotheses['H2_total_comparable_contexts'])} comparisons. These reversals are consistent with operator coverage, data conversion, memory access, and runtime-kernel differences, but the logs do not identify a single causal mechanism.

H3 was {h3}: {int(hypotheses['H3_dominated_int8_configurations_median_front'])} INT8 configuration rows were outside the median-based Pareto front, so an unconditional “always INT8” rule would retain choices dominated under the frozen validation/size/device evidence. The poor MobileNetV3-Small full-INT8 parity and its XNNPACK configuration failures show why integer-only conversion success is not equivalent to an accurate, executable deployment [@P01; @P09; @P29]. This agrees with the broader Android finding that quantization outcomes depend on model and execution path [@P33].

For H4, the median trial-level CV was {', '.join(f'{key}={float(value):.4f}' for key, value in sorted(cv_values.items()))}, a max/min ratio of {float(hypotheses['H4_max_to_min_runtime_cv_ratio']):.2f}. These descriptive differences and the median-versus-p95 frontier change are consistent with H4, but {int(summary['complete_trials_per_successful_configuration'])} complete trials on one device are insufficient for a population-level significance claim. Tail summaries therefore add decision information without converting repeated invocations into pseudo-replicates.

Protocol 1.1 produced an incomplete ANR-terminated run when synchronous LiteRT calls occupied Flutter's UI isolate. That raw file was hashed, preserved, and categorically excluded. Protocol 1.2 transferred the frozen flatbuffers to a background Dart isolate and restarted all trials from the beginning. This correction is part of the failure provenance, not a pooled pilot observation."""

    shared_recommendations = sorted(front_overlap or median_front or p95_front)
    conclusion_body = f"""On the tested realme RMX3760 stack, quantization label and file size alone did not determine deployment performance. The 12 frozen flatbuffers spanned {pct(models['accuracy'].min())}-{pct(models['accuracy'].max())} locked-test accuracy and {mib(models['artifact_size_mb'].min())}-{mib(models['artifact_size_mb'].max())} MiB, while runtime/thread choices changed latency order and median-versus-p95 criteria changed the Pareto frontier. Configuration errors and the MobileNetV3-Small full-INT8 quality loss remained part of the result rather than being hidden by fallback or post-test exclusion.

For this phone and software stack only, a practical shortlist is {front_text(set(shared_recommendations))}; selection among it should use the application's tolerance for median versus tail latency together with the frozen validation quality and artifact-size constraints. No cross-device recommendation follows from one phone, and a new device/runtime combination should rerun the same hashed, complete-trial protocol."""

    text = args.manuscript.read_text(encoding="utf-8")
    abstract_results = (
        f"Locked-test accuracy ranged from {pct(models['accuracy'].min())} to {pct(models['accuracy'].max())} "
        f"and artifacts from {mib(models['artifact_size_mb'].min())} to {mib(models['artifact_size_mb'].max())} MiB; "
        f"the fastest mean trial median and p95 were {ms(fastest_median['trial_median_mean_ms'])} ms "
        f"({fastest_median['configuration_id']}) and {ms(fastest_p95['trial_p95_mean_ms'])} ms "
        f"({fastest_p95['configuration_id']}), respectively. Trial-pair rank correlations had minimum "
        f"{float(summary['rank_stability_spearman_min']):.3f} and median "
        f"{float(summary['rank_stability_spearman_median']):.3f}; "
        f"{int(audit['failed_configurations'])} configurations produced "
        f"{int(audit['configuration_error_rows'])} preserved error rows."
    )
    text, count = re.subn(
        r"\*\*\[FINAL RESULTS SENTENCE:.*?error count\.\]\*\*",
        abstract_results,
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise ValueError("Abstract results placeholder not found")
    text = text.replace(
        "Median- and\np95-based Pareto fronts **[FINAL PARETO RESULT]**.",
        f"Median- and p95-based Pareto fronts contained {len(median_front)} and {len(p95_front)} configurations, with {len(front_overlap)} shared.",
    )
    text = text.replace(
        "The findings demonstrate\n**[FINAL CONSERVATIVE CONCLUSION]** on the tested realme RMX3760;",
        "The findings demonstrate that compression did not guarantee lower latency and that runtime and tail criteria changed configuration ranking on the tested realme RMX3760;",
    )
    text = replace_section(text, "## 4.1 Training, conversion, and locked-test classification", "## 4.2 Android latency, memory, and operational failures", classification_body)
    text = replace_section(text, "## 4.2 Android latency, memory, and operational failures", "## 4.3 Rank stability and Pareto sensitivity", android_body)
    text = replace_section(text, "## 4.3 Rank stability and Pareto sensitivity", "# 5. Discussion", rank_body)
    text = replace_section(text, "# 5. Discussion", "# 6. Threats to validity and limitations", discussion_body)
    text = replace_section(text, "# 7. Conclusion", "# Declarations", conclusion_body)
    text = text.replace(
        "Its experiment screen runs the\nfrozen randomized benchmark and writes an immutable CSV to app-specific\nexternal storage.",
        "Its experiment screen runs the frozen randomized benchmark in a background Dart isolate, with frozen flatbuffers transferred from the root isolate, and writes an immutable CSV to app-specific external storage. This keeps the UI responsive without changing the interpreter-only timed region.",
    )
    if any(marker in text for marker in ("[FINAL", "[AUTO-FILL")):
        raise ValueError("Scientific result placeholder remains after finalization")
    args.manuscript.write_text(text, encoding="utf-8")
    print(json.dumps({
        "status": "MANUSCRIPT_RESULTS_POPULATED",
        "manuscript": str(args.manuscript),
        "model_rows": len(model_rows),
        "configuration_rows": len(config_rows),
        "median_front_count": len(median_front),
        "p95_front_count": len(p95_front),
    }, indent=2))


if __name__ == "__main__":
    main()
