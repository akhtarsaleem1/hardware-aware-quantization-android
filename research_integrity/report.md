# Research integrity report

Updated: 2026-08-24

## Actually completed

- Verified 34 real literature records and 34 unique bibliography entries; all
  manuscript citation keys resolve.
- Validated all 17,509 DeepWeeds images and froze a capture-session-grouped
  10,506/3,502/3,501 split with no unresolved cross-split candidate.
- Trained MobileNetV2, MobileNetV3-Small, and EfficientNet-B0 under a
  fail-closed TensorFlow 2.21 WSL2 policy on the NVIDIA GeForce GTX 1650.
- Exported and hash-verified FP32, FP16, dynamic INT8, and full INT8
  flatbuffers using one 800-sample training-only calibration artifact.
- Built and tested a local-only Flutter research prototype and release
  benchmark APK (protocol 1.2.0, app 1.2.0+3).
- Completed and independently audited a post hoc bounded Android dataset with
  two complete balanced trials, 13,800 measured rows,
  69 successful configurations, and
  6 explicit error rows on
  realme_RMX3760_api35.
- Froze validation/device Pareto selection before opening the test set, then
  performed one locked 3,501-image final-test session with Keras references
  verified on /GPU:0.
- Generated hash-linked tables, high-resolution/vector figures, a populated
  manuscript, supporting statements, and 118 provenance rows.

## Failed experiments and negative evidence

- EfficientNet-B0 batch 16 produced a recorded GPU ResourceExhaustedError;
  batch 8 was frozen only after all architectures passed GPU feasibility.
- MobileNetV3-Small full INT8 had poor validation parity and was retained.
- Protocol 1.1 ended in an ANR after synchronous LiteRT work occupied the UI
  isolate. Its 3,738-row partial CSV is hashed and
  excluded.
- A protocol-1.2 attempt was paused when another app took foreground; Android
  killed PID 25813 for
  Watermark Pressure. Its 3,865-row partial CSV is
  hashed and excluded. It was never resumed, pooled, or imputed.
- A second protocol-1.2 candidate lost the foreground condition to
  com.whatsapp/.Conversation. Its 1,886-row
  partial CSV is hashed and excluded; the process was stopped without reuse.
- The final protocol-1.2 candidate stopped during trial 3. Its
  17,168-row source CSV is hashed and retained;
  trial 3 was excluded wholesale, with no pooling or imputation. Only balanced
  trials 1-2 enter the explicitly exploratory bounded analysis.
- Final configuration failures remain explicit; no fallback measurements were
  introduced.

## Not completed or not claimed

- NNAPI was excluded because effective execution could not be verified on this
  Android 15 stack and the available API path was deprecated.
- Energy use was not attributed per model; no cold-start claim is made.
- No second physical phone was available, so cross-device generalization and
  hardware-population significance are not claimed.
- No direct live Clarivate profile was captured; SCIE indexing is not claimed.
- Author metadata, declarations, sole-author approval, and a reserved Zenodo
  DOI were supplied on 2026-08-24. The Zenodo record still requires publication
  before journal submission.

## Assumptions, bias risks, and limitations

- One phone, two CPU paths, one dataset, three architectures, and post-training
  quantization limit generalization.
- USB charging and uncontrolled background apps are recorded confounders.
- PSS/RSS are observational process snapshots, not isolated peak allocations.
- Two complete balanced trials support only descriptive repeatability and
  imprecise device-level uncertainty; the planned third trial was excluded
  wholesale after the run became incomplete. This post hoc bounded analysis is
  exploratory, not confirmatory, and invocations are not independent device
  replicates.
- Pareto quality uses frozen validation accuracy; locked-test metrics were not
  used to choose configurations.

## Reproducibility status

Scientific-package reproducibility is complete for the recorded workstation and
phone stack. The audited raw CSV SHA-256 is 5c129b5d6adfa10e3db3f4f66e118665470e89ceffd780d414dcc06483dd762b; the
locked prediction CSV SHA-256 is bd711a6f0f97182a01ead454dd60bcf305f301c9e1e328c6518e02e51a30be30. Every
supplement artifact is hashed in its supplement manifest.
