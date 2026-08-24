# Repeatable Quantization and Runtime Benchmarking on Android

Status: **publication package complete; GitHub repository prepared; Zenodo publication and live journal-index verification remain administrative checks**

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
- Android matrix: 13,800 observations,
  69 successful configurations, and
  6 preserved error rows.
- Locked test: one session over 3,501 images after
  validation/device selection was frozen.
- Best locked-test accuracy: 84.43% for
  mobilenet_v2 float32.
- Fastest mean complete-trial median:
  146.77 ms for
  mobilenet_v3_small_full_int8__builtin_cpu__4t.
- Supplement: 9 tables and
  7 vector figures with hashes.

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
- 12 exact model assets under flutter_app/; the 126.7 MB release APK is distributed in the Zenodo artifact because it exceeds GitHub's single-file limit

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
https://doi.org/10.5281/zenodo.22080295 must be published before journal upload.
No SCI/SCIE indexing claim is made; verify the live Master Journal List and the
current Guide for Authors immediately before submission.

Publication gate: READY_FOR_AUTHOR_SUBMISSION_REVIEW.

