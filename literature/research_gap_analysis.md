# Candidate research gaps and provisional selection

Updated: 2026-08-21

## What is already well established

- Quantization and hardware-aware mixed-precision search are mature research
  areas; neither can be claimed as novel.
- Real-smartphone AI benchmarks and latency-prediction systems already exist.
- Model size, FLOPs, and parameter count can fail to predict measured latency.
- Runtime kernels, graph partitioning, memory traffic, unsupported operations,
  and precision conversions affect realized acceleration.
- Pareto methods are already used for accuracy-efficiency analysis.

## Gap A: auditable rank stability of ordinary Android configurations

**Question.** For unchanged lightweight classifiers, does the ranking of FP32,
FP16, dynamic INT8, and full INT8 remain stable across LiteRT delegate and
thread settings on a consumer Android phone?

**Evidence.** Hardware-aware quantization work commonly searches custom
per-layer bit widths or architectures. Broad mobile benchmarks compare
hardware/software stacks. The reviewed set contains little combined evidence
for the practitioner-facing matrix of standard exported variants, delegates,
threads, repeated complete trials, raw tail latency, and explicit fallback
metadata.

**Why it matters.** This is the decision actually faced when an application can
bundle several ordinary TFLite/LiteRT artifacts. A silent delegate fallback can
reverse a ranking while file size remains unchanged.

**Feasibility.** High on one laptop and the connected realme RMX3760. Three
architectures by four precisions can be reduced after conversion/parity pilots.
The app and analysis are manageable without a hardware laboratory.

**Cost.** Moderate training/conversion cost and potentially many short phone
benchmark blocks. Heavy training will use the NVIDIA GPU when verified.

**Likely contribution.** Reproducible measurement and transparent selection
methodology rather than a new quantizer.

**Publication risk.** Medium. The novelty is incremental and one-phone evidence
limits generalization. Risk can be reduced through unusually strong raw-data,
fallback, repeatability, and provenance controls.

## Gap B: transferability of a selector across heterogeneous Android phones

**Question.** Can device descriptors predict the best precision/runtime
configuration on an unseen phone?

**Evidence.** nn-Meter, MobileNetV4, AI Benchmark, and MLPerf Mobile establish
strong device dependence and latency prediction precedent. A configuration
recommender across commodity Android phones remains useful.

**Why it matters.** It would enable deployment-time configuration choice before
exhaustive benchmarking on every handset.

**Feasibility.** Low with one reliably connected phone. The previous study has
aggregate evidence from five phones but not the required precision/runtime
matrix or raw observations, so it cannot train or validate this selector.

**Cost.** High. It requires several phones, matched experiment blocks, and
enough devices to hold out hardware families.

**Likely contribution.** Stronger than Gap A if sufficient devices exist.

**Publication risk.** High under current resources because a model trained on a
handful of phones would overfit and unsupported generalization claims would be
tempting.

## Gap C: sustained thermal and energy effects on quantization ranking

**Question.** Does the preferred configuration change during sustained
inference as temperature and power-management state evolve?

**Evidence.** The reviewed papers report energy or hardware utilization
selectively, while many headline latency results are short runs. The connected
phone exposes Thermal HAL telemetry, making thermal context observable.

**Why it matters.** Short cold-start benchmarks may not represent sustained
camera or streaming workloads.

**Feasibility.** Medium for thermal analysis; low for defensible energy
measurement without an external power monitor or reliable rail telemetry.

**Cost.** High device time and careful cooldown/control protocol.

**Likely contribution.** A valuable ablation or follow-up study.

**Publication risk.** Medium-high because uncontrolled background activity and
indirect battery estimates can undermine the result.

## Provisional selection after the 2025 novelty conflict

The broad form of Gap A is no longer defensible as novel because Gherasim and
García Sánchez (2025) already compare model families, quantizations, CPU/GPU/NPU
execution, CPU threading, accuracy, average latency, and Pareto fronts on an
Android tablet. Gap A is narrowed to **distributional repeatability and rank
stability**. Gap C's thermal context becomes part of that reproducibility study,
but no energy claim is planned. Gap B remains deferred until at least five
phones can run the full repeated raw-data matrix.

The selection is provisional until a focused 2025-2026 novelty audit is
complete.

## Revised working title

> Beyond Average Latency: Repeatability and Runtime-Dependent Quantization
> Rankings for Lightweight Vision Models on Android

## Revised research questions

- **RQ1:** What accuracy and artifact-size changes result from FP32, FP16,
  dynamic-range INT8, and full INT8 conversion for each selected architecture?
- **RQ2:** How do precision, effective delegate, and thread count interact in
  median and tail latency on the connected Android phone?
- **RQ3:** Is configuration ranking stable across repeated complete benchmark
  trials and recorded thermal contexts?
- **RQ4:** Are the smallest artifacts also the lowest-latency configurations?
- **RQ5:** Which configurations remain Pareto-efficient when median latency is
  replaced by p95 latency or when cross-trial uncertainty is considered?
- **RQ6:** How often do conversion, delegate initialization, partitioning, or
  fallback behavior prevent the theoretically preferred precision from being
  the operationally preferred configuration?

## Revised hypotheses

- **H1:** File-size ranking and median-latency ranking are not identical across
  all tested configurations.
- **H2:** At least one architecture exhibits a precision-by-runtime interaction
  large enough to change the latency ranking.
- **H3:** A Pareto selector using measured validation/configuration evidence
  avoids at least one dominated choice made by the fixed rule "always INT8."
- **H4:** Within-configuration variance differs across runtime/delegate settings,
  so median latency alone does not fully describe operational stability.

H1-H4 remain untested. They must be supported, rejected, or left inconclusive
from executed experiments.
