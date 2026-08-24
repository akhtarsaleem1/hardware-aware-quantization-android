# Novelty audit

Updated: 2026-08-22

## Critical overlap

**Iulius Gherasim and Carlos García Sánchez. Hardware optimization on Android
for inference of AI models. arXiv:2511.13453, 2025.**

Verified overlap from the full text:

- Android execution on a stock Samsung Galaxy Tab S9;
- ResNet image classification and YOLO object detection;
- CPU single/multicore, GPU FP32/FP16, and NPU execution;
- FP32, FP16, INT8, full INT8, INT16, full INT16, and dynamic variants;
- latency and accuracy comparisons;
- model/runtime conversion effects;
- accuracy-latency Pareto analysis and recommended configurations.

This paper substantially covers the original broad contribution. It must be
cited prominently and prevents claims that this project introduces Android
hardware-aware quantization comparison or Pareto-based configuration choice.

## Narrowed extension

The reviewed paper reports average inference time across a validation dataset.
Its presented methodology does not establish raw per-inference distribution,
tail percentiles, repeated randomized complete blocks, cross-trial rank
stability, thermal-state relationships, process-memory distributions, or
statistical/effect-size comparisons. It uses a high-end Snapdragon tablet and
ResNet/YOLO rather than low-end phone hardware and MobileNet-family classifiers.

The new study therefore targets:

1. repeatability and rank stability rather than another average-latency table;
2. immutable raw latency vectors and distributional statistics;
3. lightweight architecture-by-precision-by-runtime interactions;
4. delegate initialization/partition/fallback provenance;
5. thermal and memory context on a lower-end consumer smartphone;
6. a replication/extension framing rather than a novelty-first framing.

## Final 2025-2026 recheck

Final search date: 2026-08-22.

The recheck covered Android/smartphone quantization selection, LiteRT hardware
benchmarking, Pareto configuration choice, p95/tail latency, repeatability,
thermal context, and rank stability. It reconfirmed Gherasim and Garcia
Sanchez (2025) as the closest direct overlap. Other 2025-2026 primary papers
found in the recheck concerned mobile-GPU generative inference, LiteRT LLMs on
Raspberry Pi/mobile devices, challenge-specific quantized super-resolution,
hardware-aware feature extraction, or latency-aware pruning. None reported the
same unchanged-classifier, runtime/thread, complete-trial distributional
protocol on Android.

This is a search result, not proof that no related work exists. The contribution
therefore remains a cautious independent replication/extension centered on raw
distributions, tail criteria, trial-rank stability, process-memory observations,
thermal/failure provenance, and a lower-cost phone.

Primary recheck sources included the arXiv record for Gherasim and Garcia
Sanchez (2511.13453) and the official CVF open-access proceedings for the
2025-2026 mobile/edge papers reviewed.

## Novelty status

`FINAL_RECHECK_COMPLETE_REPLICATION_EXTENSION_FRAMING`

No claim is made that Android hardware-aware quantization comparison or
Pareto-based selection is new.

