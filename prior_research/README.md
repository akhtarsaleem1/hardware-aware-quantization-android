# Prior plant-disease study transfer

This directory records what the new study learned from
`../plant_disease_research` (a sibling project in the workspace). It is a
methodological input and provenance record, not a source of new experimental
results.

## Reused concepts

- FP32, FP16, dynamic-range INT8, and full INT8 conversion attempts.
- Representative training data for full-integer calibration.
- Input quantization and output dequantization using tensor scale/zero point.
- Prediction-parity evaluation against the Keras reference.
- Warm-up followed by repeated on-device inference.
- Release-mode physical-device checks.
- Explicit recording of conversion/delegate failures.

## Important prior observations

On the earlier MobileNetV3-Small plant classifier, the saved comparison shows:

- FP32: 11.478 MiB, test accuracy 0.96858, desktop median 10.732 ms;
- FP16: 5.777 MiB, test accuracy 0.96809, desktop median 10.696 ms;
- dynamic range: 3.159 MiB, test accuracy 0.96195, desktop median 36.551 ms;
- full INT8: 3.359 MiB, test accuracy 0.72754, desktop median 810.258 ms after
  the default XNNPACK delegate failed and execution retried with built-ins.

These figures demonstrate a motivating counterexample to "smaller is faster"
in that particular prior environment. They do **not** establish the hypotheses
of the new project because model, dataset, host runtime, delegate behavior, and
measurement scope differ.

The prior Android work benchmarked only the selected FP16 model across five
phones. It did not compare quantization modes on each phone, did not retain raw
per-inference measurements, and usually had one aggregate trial per phone.
Those limitations directly shape the new design.

## Deliberately not copied

- trained plant-disease models and labels;
- field images or predictions;
- old figures as new paper assets;
- old manuscript wording;
- aggregate Android rows in the new `results/` directory.

See `artifact_inventory.csv` for source hashes and `lessons_learned.md` for the
design changes.

