# Experimental design gate

Status: **provisional; must be finalized after literature review**

## Core factors

The initial feasibility matrix contains three lightweight architectures,
four numeric configurations, up to three runtime/delegate modes, and three
thread settings. It is a design space, not a commitment to execute every
combination.

| Factor | Provisional levels | Decision gate |
|---|---|---|
| Architecture | MobileNetV2, MobileNetV3-Small, one literature-justified third model | Dataset and literature review |
| Precision | FP32, FP16, dynamic-range INT8, full INT8 | Successful conversion and parity check |
| Runtime | LiteRT CPU/XNNPACK, supported GPU/vendor delegate, NNAPI legacy-only | Verified effective execution on device |
| Threads | 1, 2, 4 | Device-appropriate feasibility pilot |
| Trials | at least 3 complete randomized blocks | Variance pilot and time budget |
| Per-trial runs | 20 warm-ups + 100 measured | Warm-up ablation |

## Fairness controls

1. Use identical test examples and preprocessing for variants of an
   architecture.
2. Build full INT8 calibration data from training data only.
3. Freeze the experiment matrix before confirmatory benchmark collection.
4. Randomize configuration order within a complete block and save the seed.
5. Record build mode, delegate, thread count, battery state, thermal status,
   screen policy, charging state, and relevant background-load policy.
6. Save one raw row per inference; never transcribe aggregate screenshots into
   confirmatory raw data.
7. Record unsupported conversions and delegate failures rather than silently
   dropping them.
8. Separate exploratory pilots from confirmatory measurements.
9. Treat NNAPI as a legacy Android-15 comparison only; Android deprecated it in
   API 35, so it is not the primary future-facing acceleration path.

## Primary outcomes

- classification accuracy and macro F1;
- median and p95 on-device latency;
- model file size;
- process-memory change where a defensible Android method is implemented.

Energy is optional and must remain `NOT_MEASURED` unless a repeatable,
device-appropriate method is documented.

## Selection protocol

The default analysis finds non-dominated configurations over accuracy
(maximize), latency (minimize), memory (minimize), and file size (minimize).
Any scalar score is a sensitivity analysis across declared weighting scenarios,
not a single arbitrarily weighted ground truth. Selection development must use
validation/calibration evidence; the held-out test set is reserved for final
evaluation.

## Stop conditions

- Do not train before the dataset license, split policy, and leakage checks are
  documented.
- Do not benchmark a converted model before numerical I/O and prediction parity
  are verified.
- Do not write a Results section before raw evidence exists.
- Do not claim hardware-general selection from one phone.
