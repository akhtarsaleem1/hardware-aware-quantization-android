# Model selection

Updated: 2026-08-21

## Core architectures

1. **MobileNetV2** — widely supported inverted-residual mobile baseline with an
   official Keras implementation and ImageNet weights.
2. **MobileNetV3-Small** — hardware-aware mobile CNN with squeeze-and-excitation,
   hard-swish, and 5 x 5 operations that may interact differently with
   delegates and quantization.
3. **EfficientNet-B0** — compact compound-scaled CNN with a distinct
   operator/memory profile and official Keras implementation.

All use 224 x 224 RGB input, the same training/validation/test allocation, the
same augmentation family, and equivalent head/fine-tuning policy. Preprocessing
is model-specific but must be embedded in or exactly documented with the model.

## Feasibility gate

Before full training, instantiate each ImageNet backbone and run:

- one GPU forward/backward batch;
- a one-epoch small-subset smoke test;
- FP32 conversion;
- FP16 and dynamic-range conversion;
- full-INT8 conversion with training-only calibration samples;
- desktop prediction parity on a fixed validation subset.

A model is removed from the confirmatory matrix only for a recorded technical
failure that cannot be fixed without changing the architecture or introducing
custom runtime operations. Negative results remain in conversion status files.

## Deferred models

MobileNetV4 is reserved for follow-up work because its current Keras and LiteRT
conversion path would require extra implementation validation. MobileViT is
deferred because transformer nonlinearities and delegate support could turn the
study into a custom quantization project. EfficientNet-Lite0 is not selected
because no equally mature official Keras application path is available.

