#!/usr/bin/env python3
"""Generate result-grounded publishing statements with supplied author metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results/final"
    summary = json.loads((results / "analysis_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "reports/final_benchmark_audit.json").read_text(encoding="utf-8"))
    test = json.loads((results / "final_test_report.json").read_text(encoding="utf-8"))
    models = pd.read_csv(results / "table_model_results.csv")
    configs = pd.read_csv(results / "table_configuration_results.csv")
    if summary.get("status") != "FINAL_ANALYSIS_COMPLETE":
        raise ValueError("Final analysis is not complete")
    if audit.get("status") != "PASS":
        raise ValueError("Benchmark audit is not PASS")
    if test.get("status") != "FINAL_TEST_SINGLE_SESSION_COMPLETE":
        raise ValueError("Final test report is not complete")

    evidence = summary["hypothesis_evidence"]
    median_front = set(summary["median_pareto_configuration_ids"])
    p95_front = set(summary["p95_pareto_configuration_ids"])
    fastest_median = configs.loc[configs["trial_median_mean_ms"].idxmin()]
    fastest_p95 = configs.loc[configs["trial_p95_mean_ms"].idxmin()]

    highlights = [
        "A hash-linked Android protocol compares 12 LiteRT variants in 72 settings.",
        "Complete trials retain median, p95, rank stability, memory, and failures.",
        f"Size and latency orders differed in {int(evidence['H1_size_vs_latency_rank_mismatch_contexts'])} of {int(evidence['H1_total_contexts'])} contexts.",
        f"Runtime changed quantization rank in {int(evidence['H2_runtime_rank_change_contexts'])} of {int(evidence['H2_total_comparable_contexts'])} contexts.",
        f"Median and p95 Pareto fronts shared {len(median_front & p95_front)} configurations.",
    ]
    too_long = [value for value in highlights if len(value) > 85]
    if too_long:
        raise ValueError(f"Highlight exceeds 85 characters: {too_long}")
    (root / "paper/highlights.md").write_text(
        "# Highlights\n\n" + "\n".join(f"- {value}" for value in highlights) + "\n",
        encoding="utf-8",
    )

    cover = f"""# Cover letter

24 August 2026

Editor-in-Chief  
Journal of Systems Architecture

Dear Editor,

Please consider our original research manuscript, **“Beyond Average Latency: Repeatability and Runtime-Dependent Quantization Rankings for Lightweight Vision Models on Android,”** for publication in the *Journal of Systems Architecture*.

The manuscript presents an auditable, system-level study of 12 deployable LiteRT flatbuffers across 72 planned model/runtime/thread configurations on a lower-cost Android 15 phone. A post hoc bounded analysis retains two complete balanced trials after the incomplete third trial was excluded wholesale, so the Android evidence is explicitly exploratory rather than confirmatory. The contribution is deliberately narrower than another application comparison: it focuses on capture-session leakage control, a test split held closed until device selection was frozen, immutable per-inference observations, complete randomized trials, p95/p99 latency, rank stability, memory and thermal context, and explicit preservation of delegate/configuration failures.

The locked 3,501-image evaluation produced {pct(models['accuracy'].min())}-{pct(models['accuracy'].max())} accuracy. The fastest mean trial median and p95 were {float(fastest_median['trial_median_mean_ms']):.2f} ms and {float(fastest_p95['trial_p95_mean_ms']):.2f} ms. File-size and median-latency order differed in {int(evidence['H1_size_vs_latency_rank_mismatch_contexts'])}/{int(evidence['H1_total_contexts'])} contexts, runtime changed quantization rank in {int(evidence['H2_runtime_rank_change_contexts'])}/{int(evidence['H2_total_comparable_contexts'])} comparisons, and median versus p95 criteria changed the Pareto frontier. {int(audit['failed_configurations'])} failed configurations were retained as evidence rather than replaced with fallback measurements.

The work fits the journal's published scope in embedded/mobile systems and software/system architecture by examining how deployable model artifacts interact with runtime kernels, delegates, threading, and measurement design. We make no unverified claim about Clarivate indexing; that administrative check remains separate from the scientific submission package.

This manuscript is original, is not under consideration elsewhere, and has not been published previously. The sole author has approved the submission. Funding and conflict-of-interest declarations are included in the manuscript. The artifact package contains code, manifests, model and raw-data hashes, frozen selection evidence, per-sample predictions, tables, figures, the release APK, and an explicit record of an excluded incomplete ANR run. A Zenodo record has been reserved at https://doi.org/10.5281/zenodo.22082237 and will be made public before manuscript submission.

Sincerely,

Akhtar Saleem  
Independent Researcher, Peshawar, Pakistan  
akhtarsaleem974@gmail.com  
ORCID: https://orcid.org/0009-0005-9440-224X
"""
    (root / "paper/cover_letter_draft.md").write_text(cover, encoding="utf-8")

    target = """# Journal-target manuscript

Primary scope target: **Journal of Systems Architecture** (Elsevier), original research article. Akhtar Saleem confirmed this target on 24 August 2026.

The journal's official scope explicitly includes embedded and mobile systems and emphasizes software/system architecture. The manuscript is framed as a reusable systems measurement contribution rather than an application-only case study. A direct live Clarivate Master Journal List profile was not available to this workflow, so the package makes no SCIE claim. Before submission, the author should confirm the current journal status using both ISSNs in the live Master Journal List and recheck the current Guide for Authors.

Prepared title: *Beyond Average Latency: Repeatability and Runtime-Dependent Quantization Rankings for Lightweight Vision Models on Android*.

Scientific package version: protocol 1.2.0; app 1.2.0+3; one locked-test session; one physical realme RMX3760 device.
"""
    (root / "paper/journal_target_version.md").write_text(target, encoding="utf-8")

    data = f"""# Data availability

DeepWeeds is available under CC BY 4.0 from its official repository. The submission artifact contains the frozen capture-session-grouped manifests and hashes, training-only calibration indices, validation/parity reports, the balanced protocol-1.2 trials 1-2 Android CSV (SHA-256 `{audit['raw_csv_sha256']}`) plus immutable provenance for the incomplete source run, the locked-test per-sample prediction CSV and aggregate JSON, derived tables, and publication figures. Source images are not duplicated beyond what the DeepWeeds license and submission venue permit. A Zenodo record has been reserved for the reproducibility artifact at https://doi.org/10.5281/zenodo.22082237 and will be made public before manuscript submission.
"""
    (root / "paper/data_availability.md").write_text(data, encoding="utf-8")

    code = """# Code availability

The artifact package includes dataset validation and grouped splitting, GPU-fail-closed training, post-training conversion, parity and flatbuffer checks, Flutter local inference and release benchmarking, independent raw-run audit, pretest Pareto selection, one-shot locked-test evaluation, statistical analysis, figure generation, manuscript population, and publication-readiness verification. Exact environment reports, APK/model/data hashes, and protocol files are included. The reproducibility artifact has the reserved Zenodo DOI https://doi.org/10.5281/zenodo.22082237; the record will be made public before manuscript submission.
"""
    (root / "paper/code_availability.md").write_text(code, encoding="utf-8")

    reproducibility = f"""# Reproducibility statement

The study is reproducible from hash-linked artifacts and fail-closed scripts. DeepWeeds was split into 10,506/3,502/3,501 training/validation/test images using deterministic capture-session grouping; the test manifest remained unopened until the device benchmark, independent audit, validation-only Pareto selection, and final-test manifest were frozen. Heavy model work executed on the recorded NVIDIA GPU; the final Keras test references also require `/GPU:0`. TFLite comparison inference uses the explicitly documented CPU runtime.

Three architectures were exported to FP32, FP16, dynamic-range INT8, and full INT8 using one 800-sample training-only calibration artifact. The Android matrix planned two runtimes, 1/2/4 threads, three randomized complete trials, 20 warm-ups, and 100 timed invocations. The run became incomplete during trial 3; trial 3 was excluded wholesale and the post hoc bounded analysis retains two complete balanced trials. Raw rows preserve model hashes, delegate state, telemetry, and errors. The analyzed bounded dataset has {int(audit['measured_rows']):,} measured observations and {int(audit['configuration_error_rows'])} error rows. Trial summaries - not individual invocations - are resampled for descriptive uncertainty, and the Android evidence is exploratory rather than confirmatory. All figures are generated at 600 dpi in PNG and vector PDF form.

Protocol 1.1 ended in a UI-isolate ANR and is retained under `benchmark/failed_runs` solely as excluded provenance. Protocol 1.2 transfers flatbuffers to a background Dart isolate and restarts the entire matrix. The publication verifier refuses scientific placeholders, missing result artifacts, broken hashes, selection/test-order violations, or a missing independent benchmark audit.
"""
    (root / "paper/reproducibility_statement.md").write_text(
        reproducibility, encoding="utf-8"
    )
    print(json.dumps({
        "status": "PUBLICATION_TEXT_FINALIZED",
        "highlights": highlights,
        "median_front_count": len(median_front),
        "p95_front_count": len(p95_front),
    }, indent=2))


if __name__ == "__main__":
    main()
