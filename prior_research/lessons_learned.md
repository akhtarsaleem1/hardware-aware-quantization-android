# Lessons carried into the new protocol

## 1. Artifact size is not a latency proxy

The dynamic-range prior artifact was the smallest successful high-accuracy
variant but ran slower than FP16 and FP32 on the inspected desktop runtime.
Therefore file size and latency remain separate measured outcomes.

## 2. "INT8" is not one operational condition

Dynamic-range and full-integer conversion produced different numerical I/O,
accuracy, delegate behavior, and latency. The new metadata must record exact
conversion flags, supported operations, calibration source, tensor types,
scale/zero point, runtime, and fallback behavior.

## 3. Delegate failure is a result

The prior full-INT8 artifact triggered an XNNPACK preparation failure. Silent
fallback would make comparison ambiguous, so the new Android engine must expose
requested and effective delegate plus initialization errors.

## 4. Aggregate-only Android measurements are insufficient

The old Flutter benchmark returned mean, median, minimum, and maximum but not
the raw latency vector or tail percentiles. The new engine must export one row
per inference and calculate summaries offline from immutable raw data.

## 5. Repetition must occur at two levels

One 100-inference loop estimates within-trial variation; it does not estimate
between-trial changes caused by temperature, scheduling, or background state.
The new design uses repeated complete randomized blocks.

## 6. Release mode and execution context must be explicit

The earlier debug and release results were not directly comparable and Android
conditions were partly qualitative. Build mode, charging, thermal status,
battery saver, screen policy, background policy, delegate, and threads are now
required fields.

## 7. Calibration and parity need first-class provenance

Full INT8 had a major accuracy loss in the prior study. The new project must
distinguish expected quantization degradation from calibration/preprocessing or
unsupported-operation problems by retaining calibration IDs and per-variant
parity evidence.

