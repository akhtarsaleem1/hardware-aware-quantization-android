# Literature review: verified first pass

Updated: 2026-08-21

## Scope and status

The database currently contains 34 publications: 29 peer-reviewed conference
papers and five clearly labelled preprints. Twenty-four are classified as
directly relevant to mobile benchmarking, deployable quantization, hardware-
aware selection, or the relation between model structure and measured latency.
The remainder provide supporting evidence on mobile architectures, memory, or
specialized on-device systems.

Metadata and technical summaries were checked against primary proceedings,
PMLR, OpenReview, IJCAI, arXiv, or the authors' official project repository.
DOIs are included only where directly verified. Empty DOI cells do not mean a
DOI does not exist. Journal indexing and 2025-2026 novelty checking remain
separate pending tasks.

## 1. Quantization does not guarantee realized acceleration

Jacob et al. established an integer-only inference path and demonstrated that
deployable INT8 can improve the latency-accuracy trade-off on ARM CPUs. Later
work clarified why a nominally quantized graph may not realize this benefit.
HAWQ-V3 explicitly removes float conversions and integer division from the
graph, while PikeLPN reports that unquantized elementwise operations can
dominate the cost of otherwise low-precision networks. I-ViT and FQ-ViT solve
similar problems for transformer nonlinearities.

This distinction matters for the proposed study: the label `INT8` is
insufficient. Conversion parameters, input/output types, unsupported operators,
delegate partitioning, and actual fallback paths must be recorded.

## 2. Optimal precision is hardware-dependent but prior optimization targets differ

HAQ, HAWQ, HAWQ-V3, QBitOpt, QuantNAS, and the differentiable mixed-precision
method of Schaefer et al. all show that layer precision can be treated as an
optimization variable. HAQ uses accelerator-simulator feedback; HAWQ uses
second-order sensitivity; HAWQ-V3 combines hardware constraints with integer-
only execution; QuantNAS searches a quantized supernet using Kirin 9000 mobile
CPU latency; and Schaefer et al. optimize an accuracy-memory Pareto frontier.

These contributions are more sophisticated than choosing among four exported
TensorFlow Lite files. They therefore prevent a defensible claim that
"hardware-aware quantization selection" itself is novel. A narrower possible
contribution is an auditable selection protocol for standard deployable
configurations under the exact Android runtime available to practitioners.

## 3. Proxy metrics and artifact size are weak latency predictors

MobileNetV2 and MobileNetV3 helped establish efficient mobile CNN design, but
more recent architecture papers make the systems issue explicit. MobileOne
reports that FLOPs and parameter count need not correlate with latency on a
phone. FastViT attributes cost to memory access and skip connections. PikeLPN
identifies supposedly minor non-quantized operations as important low-precision
costs. MobileNetV4 evaluates architecture blocks across CPUs, DSPs, GPUs, Apple
Neural Engine, and Pixel EdgeTPU and finds device-dependent Pareto behavior.

The prior plant-disease study supplies a local motivating observation with the
same direction: its smallest dynamic-range artifact was slower on the measured
desktop runtime than FP16 and FP32. That observation is not merged into the new
results, but it justifies testing rather than assuming the ranking.

## 4. Mobile benchmarks demonstrate stack complexity

AI Benchmark and MLPerf Mobile evaluate real production smartphones and show
that mobile AI performance reflects the entire hardware-software stack.
MLPerf's run rules and transparency requirements are the most relevant
methodological baseline. nn-Meter further demonstrates that useful latency
models are device/runtime specific and models fused into execution kernels
behave differently from raw operator counts.

Ahn et al. compare FP16 and static/dynamic INT8 over OpenVINO, TensorFlow Lite,
ONNX, and PyTorch on x86 and Raspberry Pi. Their results support the interaction
between precision and optimized runtime but do not cover Android phones.
MobileAIBench examines quantization, latency, and hardware use on real iOS
devices for LLM/LMM workloads, showing that configuration-level evaluation is
also relevant beyond vision CNNs.

## 5. Gaps in measurement coverage

The matrix shows several recurring omissions:

- Hardware-aware quantization papers often use custom mixed-precision methods,
  simulators, GPUs, or one named accelerator rather than standard TFLite PTQ
  variants on consumer Android phones.
- Broad phone benchmarks compare devices and stacks, but generally do not train
  a transparent per-device recommender across architecture, quantization,
  delegate, and thread settings.
- Architecture papers report headline mobile latency but rarely retain raw
  per-inference values or study repeated complete trials under thermal context.
- Peak memory, tail latency, delegate fallback, and thread-count interactions
  are reported less consistently than mean latency or throughput.
- Few studies combine measured accuracy, artifact size, memory, and latency in
  a reproducible Pareto analysis using ordinary deployable models.

These are corpus-level observations, not proof that no such paper exists. A
focused novelty search across 2025-2026 publications is still required before
the manuscript states a gap.

## 6. Implications for model selection

MobileNetV2 remains justified as a widely supported baseline with simple
operations. MobileNetV3-Small is relevant because its hardware-aware design and
nonlinearities may interact differently with delegates and quantization. A
third model should expose a distinct operator/memory profile without making the
training matrix infeasible. MobileNetV4 is scientifically current but framework
and conversion support must be tested. EfficientNet-Lite0 is a safer fallback
because it appears directly in quantized edge work. MobileViT should be included
only if a pilot confirms full conversion and delegate support.

The final core set is MobileNetV2, MobileNetV3-Small, and EfficientNet-B0,
subject to a conversion-feasibility pilot. All three have official Keras
implementations and ImageNet weights; their inverted-residual, hardware-aware
mobile, and compound-scaled blocks provide distinct operator profiles without
introducing an unverified third-party architecture implementation. MobileNetV4
is scientifically current but is deferred because the official Keras path and
conversion behavior are not yet mature enough for the constrained main study.
Failed conversion remains evidence rather than a silently removed row.


## 7. Runtime implications in 2026

The original brief proposed CPU/XNNPACK and NNAPI. This must be revised because
Android deprecated NNAPI in Android 15. NNAPI can remain a legacy comparison on
the connected Android 15 phone only if it still executes, but it should not be
presented as the recommended future path. The primary comparison should use the
current LiteRT CPU/XNNPACK path and an actually supported GPU or vendor delegate.
Requested and effective delegates must both be logged.

## 8. Provisional conclusion

The broad idea is not novel: hardware-aware mixed precision, mobile latency
prediction, real-phone benchmarking, and Pareto optimization all have strong
precedent. The defensible direction is narrower and methodological: determine
whether ordinary deployable precision/runtime configurations change rank on a
specific consumer Android phone, preserve raw repeated evidence and fallback
metadata, and select non-dominated configurations without claiming universal
hardware generalization.

## 9. Material 2025 novelty conflict

Gherasim and García Sánchez's November 2025 preprint substantially overlaps the
original proposal. It evaluates ResNet and YOLO families on a Snapdragon 8 Gen 2
Android tablet across CPU, GPU, and NPU execution; FP32, FP16, INT8, full INT8,
INT16, full INT16, and dynamic conversion; single- and multi-core CPU modes;
accuracy; average latency; and accuracy-latency Pareto fronts. It also reports
that dynamic quantization can be slow or fail on the NPU and that full INT8
collapsed ResNet18 accuracy in their conversion pipeline.

Consequently, this project must not claim novelty for Android
precision/accelerator comparison or Pareto selection. The remaining defensible
extension is distributional and reproducibility-oriented: lightweight
MobileNet-family models on a lower-end consumer phone; immutable raw
per-inference observations; randomized repeated blocks; median and tail
latency; uncertainty and effect sizes; memory; thermal context; effective
delegate/fallback logging; and explicit rank-stability analysis.
