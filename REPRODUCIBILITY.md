# Reproducibility protocol

## Frozen identifiers

- Dataset protocol: grouped_capture_session_v1_gap90_seed42
- Seed: 42
- Training batch size: 8
- Calibration: 800 training-only samples
- Android protocol/app: 1.2.0 / 1.2.0+3
- Device: realme_RMX3760_api35
- Raw benchmark SHA-256: 5c129b5d6adfa10e3db3f4f66e118665470e89ceffd780d414dcc06483dd762b
- Prediction CSV SHA-256: bd711a6f0f97182a01ead454dd60bcf305f301c9e1e328c6518e02e51a30be30
- Complete measured rows: 13,800
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
