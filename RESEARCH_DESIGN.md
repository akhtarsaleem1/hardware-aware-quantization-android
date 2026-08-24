# Frozen research design before dataset acquisition

Updated: 2026-08-21

Status: methods frozen for the dataset feasibility stage; hypotheses untested.

## Working title

Beyond Average Latency: Repeatability and Runtime-Dependent Quantization
Rankings for Lightweight Vision Models on Android

## Research questions

- **RQ1:** What changes in held-out classification performance and artifact
  size result from FP32, FP16, dynamic-range INT8, and full INT8 conversion for
  each selected architecture?
- **RQ2:** How do precision, effective delegate, and thread count interact in
  median and tail latency on the tested Android phone?
- **RQ3:** Is configuration ranking stable across repeated randomized benchmark
  blocks and recorded thermal contexts?
- **RQ4:** Are the smallest artifacts also the lowest-latency configurations?
- **RQ5:** Which configurations remain Pareto-efficient when median latency is
  replaced by p95 latency or cross-trial uncertainty is considered?
- **RQ6:** How often do conversion failure, delegate initialization,
  partitioning, or fallback prevent a nominal precision/delegate combination
  from being operationally usable?

## Hypotheses

- **H1:** File-size rank and median-latency rank are not identical for every
  tested architecture/runtime combination.
- **H2:** At least one architecture has a precision-by-runtime interaction that
  changes the latency ranking.
- **H3:** A Pareto selector fitted only with validation/configuration evidence
  avoids at least one dominated choice made by the fixed rule `always INT8`.
- **H4:** Within-configuration latency variability differs across effective
  runtime/delegate settings, so median latency alone is insufficient.

These hypotheses will be supported, rejected, or marked inconclusive only from
executed experiments.

## Contribution boundary

The study does not claim a new quantizer, Android benchmarking in general, or
Pareto selection in general. Its intended contribution is an auditable
distributional protocol for ordinary deployable LiteRT variants: immutable raw
per-inference observations, randomized repeated blocks, tail latency,
uncertainty/effect sizes, thermal context, effective delegate/fallback records,
and rank-stability analysis on a lower-cost consumer Android phone.

The November 2025 preprint by Gherasim and Garcia Sanchez already covers broad
precision/accelerator comparisons and average-latency Pareto fronts on a
Snapdragon 8 Gen 2 Android tablet. It is therefore treated as a competing paper,
not hidden or reframed as supporting novelty.

## Frozen feasibility inputs

- Dataset: DeepWeeds, official fold 0 subject to leakage audit.
- Models: MobileNetV2, MobileNetV3-Small, EfficientNet-B0.
- Image size: 224 x 224 RGB.
- Precisions: FP32, FP16, dynamic-range INT8, full INT8 where conversion and
  parity gates pass.
- Device: realme RMX3760, Android 15, UMS9230H.
- Primary outcomes: macro F1, balanced accuracy, artifact size, median latency,
  p95 latency, and defensible process-memory change.
- Selection: Pareto analysis; no arbitrary primary scalar weights.

## Hard gates

Training cannot begin until the archive hash, license, manifest, image
integrity, class distribution, exact duplicates, near-duplicate candidates, and
capture-session overlap have been recorded. Heavy tensor work must select the
NVIDIA GPU and abort on unexpected CPU fallback. Confirmatory Android
benchmarking cannot begin until conversion parity and effective delegate
execution are verified.
