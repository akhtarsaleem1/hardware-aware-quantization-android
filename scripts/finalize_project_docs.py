#!/usr/bin/env python3
"""Replace early scaffold documentation with a completed evidence handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def require(path: Path, status: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != status:
        raise ValueError(f"{path} status is not {status}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    audit = require(root / "reports/final_benchmark_audit.json", "PASS")
    analysis = require(
        root / "results/final/analysis_summary.json", "FINAL_ANALYSIS_COMPLETE"
    )
    test = require(
        root / "results/final/final_test_report.json",
        "FINAL_TEST_SINGLE_SESSION_COMPLETE",
    )
    supplement = require(
        root / "results/final/publication_supplement/supplement_manifest.json",
        "PUBLICATION_SUPPLEMENT_COMPLETE",
    )
    readiness = json.loads(
        (root / "reports/publication_readiness.json").read_text(encoding="utf-8")
    )
    if readiness.get("status") not in (
        "SCIENTIFIC_PACKAGE_READY_AUTHOR_METADATA_REQUIRED",
        "READY_FOR_AUTHOR_SUBMISSION_REVIEW",
    ):
        raise ValueError("publication readiness gate has not passed")
    models = pd.read_csv(root / "results/final/table_model_results.csv")
    configurations = pd.read_csv(
        root / "results/final/table_configuration_results.csv"
    )
    best_accuracy = models.loc[models["accuracy"].idxmax()]
    fastest = configurations.loc[
        configurations["trial_median_mean_ms"].idxmin()
    ]

    readme = f"""# Repeatable Quantization and Runtime Benchmarking on Android

Status: **publication package complete; repository publication and live journal-index verification remain administrative checks**

Final title:

> Beyond Average Latency: Repeatability and Runtime-Dependent Quantization
> Rankings for Lightweight Vision Models on Android

This repository contains a completed, hash-linked study of FP32, FP16,
dynamic-range INT8, and full INT8 LiteRT artifacts for MobileNetV2,
MobileNetV3-Small, and EfficientNet-B0 on a realme RMX3760.

## Evidence at a glance

- DeepWeeds: 17,509 images; grouped train/validation/test split
  10,506/3,502/3,501; no unresolved cross-split candidate.
- Training: TensorFlow 2.21 under WSL2, fail-closed on NVIDIA GTX 1650 GPU.
- Artifacts: 12 frozen flatbuffers and one 800-sample training-only calibration.
- Android matrix: {int(audit['measured_rows']):,} observations,
  {int(audit['successful_configurations'])} successful configurations, and
  {int(audit['configuration_error_rows'])} preserved error rows.
- Locked test: one session over {int(test['sample_count']):,} images after
  validation/device selection was frozen.
- Best locked-test accuracy: {100 * float(best_accuracy['accuracy']):.2f}% for
  {best_accuracy['architecture']} {best_accuracy['quantization']}.
- Fastest mean complete-trial median:
  {float(fastest['trial_median_mean_ms']):.2f} ms for
  {fastest['configuration_id']}.
- Supplement: {int(supplement['table_count'])} tables and
  {int(supplement['figure_pdf_count'])} vector figures with hashes.

## Main contribution

The contribution is a reproducible distributional evaluation pipeline, not a
claim that quantization or Android benchmarking is new. The experiment preserves
tail latency, complete-trial rank stability, PSS/RSS snapshots, thermal context,
unsupported configurations, and two excluded incomplete phone runs. Pareto
selection uses frozen validation quality and device measurements; locked-test
metrics are reporting outcomes only.

## Prior plant-disease work

The earlier study supplied methodological lessons for typed tensor I/O,
conversion parity, warm-ups, repeated measurements, and negative-result
preservation. Its measurements are not pooled into this project. The exact
inventory is under prior_research/.

## Key outputs

- paper/manuscript_main.md, paper/manuscript_main.docx, and .pdf
- paper/cover_letter_draft.md and paper/highlights.md
- benchmark/raw/ and reports/final_benchmark_audit.json
- results/final/final_test_report.json and per-sample predictions
- results/final/publication_supplement/ with nine tables and PNG/PDF figures
- research_integrity/report.md and result_provenance.csv
- release APK and 12 exact model assets under flutter_app/

## Reproduce

Lightweight checks:

    python -m pip install -r requirements.txt
    python -m pytest tests -q
    cd flutter_app
    flutter analyze
    flutter test

GPU-required model work runs through the WSL launcher and aborts on unexpected
CPU fallback:

    bash scripts/run_wsl_gpu_training.sh --model all
    bash scripts/run_wsl_gpu_python.sh scripts/evaluate_final_test.py --help

The final locked-test evaluator is single-session and refuses reruns once its
exclusive start record exists. Do not execute it again in this completed
artifact.

## Submission status

Prepared for submission to the Journal of Systems Architecture. Author,
affiliation, corresponding email, funding/conflict declarations, sole-author
approval, and the reserved Zenodo DOI are populated. The Zenodo record at
https://doi.org/10.5281/zenodo.22082237 must be published before journal upload.
No SCI/SCIE indexing claim is made; verify the live Master Journal List and the
current Guide for Authors immediately before submission.

Publication gate: {readiness['status']}.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    reproducibility = f"""# Reproducibility protocol

## Frozen identifiers

- Dataset protocol: grouped_capture_session_v1_gap90_seed42
- Seed: 42
- Training batch size: 8
- Calibration: 800 training-only samples
- Android protocol/app: 1.2.0 / 1.2.0+3
- Device: {audit['device_id']}
- Raw benchmark SHA-256: {audit['raw_csv_sha256']}
- Prediction CSV SHA-256: {test['prediction_file_sha256']}
- Complete measured rows: {int(audit['measured_rows']):,}
- Complete balanced trials analyzed: 2 of 3 planned; 20 warm-ups and 100
  measured invocations per successful configuration/trial.

## Reproduction order

1. Create the environment from environment.yml or requirements.txt.
2. Verify the DeepWeeds archive and grouped manifest hashes.
3. Run dataset validation and require reports/dataset_report.json PASS.
4. Run GPU feasibility, training, and validation through WSL GPU launchers.
5. Create the calibration tensor from training identifiers only.
6. Convert and verify all 12 flatbuffers and validation parity reports.
7. Prepare Flutter assets, verify hashes, and build a release APK.
8. Run the complete randomized benchmark without leaving the phone app.
9. Run scripts/audit_final_benchmark.py.
10. Run validation/device Pareto selection and freeze the final-test manifest.
11. Run scripts/evaluate_final_test.py exactly once through the GPU launcher.
12. Generate analysis, supplement, manuscript, integrity ledger, DOCX/PDF,
    and the publication-readiness report.

## Integrity controls

- The test split stayed locked until benchmark audit and pretest selection.
- Keras training/evaluation tensors are fail-closed on /GPU:0; LiteRT
  flatbuffer comparison uses the documented CPU runtime.
- Per-inference rows are never treated as independent phone replicates.
- Partial protocol-1.1 ANR and protocol-1.2 Watermark Pressure runs are hashed,
  excluded, and never pooled or resumed.
- Unsupported configurations remain error rows; no fallback is imputed.
- Every supplemental artifact and reported result category is hash-linked.

## Scope limits

One physical phone, two CPU execution paths, one dataset, three architectures,
post-training quantization, USB charging, and a post hoc bounded analysis of two
complete trials limit generalization. The planned third trial was excluded
wholesale after the run became incomplete; the Android evidence is exploratory,
not confirmatory. NNAPI and per-model energy attribution were not defensibly measurable and are
not claimed.
"""
    (root / "REPRODUCIBILITY.md").write_text(reproducibility, encoding="utf-8")
    print(json.dumps({
        "status": "PROJECT_DOCUMENTATION_FINALIZED",
        "readiness": readiness["status"],
        "analysis_status": analysis["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
