# Phase 0 environment report

Captured on 2026-08-21 in Asia/Karachi.

## Workstation

- Windows 11 Home, build 26200, x64.
- HP Victus 15-fa0xxx.
- Intel Core i5-12450H: 8 physical cores and 12 logical processors.
- 32,436 MB physical memory reported by Windows.
- NVIDIA GeForce GTX 1650 (4,096 MiB reported by `nvidia-smi`) and Intel UHD
  Graphics.
- Approximately 105.35 GiB free on the workspace drive at capture time.

The Codex bundled Python is 3.12.13 and does not contain TensorFlow or PyTorch.
Heavy work instead uses Ubuntu 26.04 under WSL2, uv-managed CPython 3.13.15,
and TensorFlow 2.21.0. TensorFlow identifies the GTX 1650 as compute capability
7.5 and created `/GPU:0` with 2,244 MiB available to the process. A matrix and
gradient smoke test passed on that device.

Flutter 3.47.1 stable and Dart 3.13.1 are installed. ADB 1.0.41 from Android
platform-tools 37.0.0 is available. Java was not on `PATH` during inspection;
Flutter may still use a bundled JDK, which must be resolved before app builds.

## Connected Android phone

ADB reported one connected realme RMX3760:

- Android 15, API 35;
- Spreadtrum UMS9230H / board platform `ums9230`;
- ARM64 primary ABI with ARMv7 compatibility;
- 8 online CPU cores;
- 5,905,892 kB total memory reported by `/proc/meminfo`;
- physical display 720 x 1600;
- Thermal HAL 2.0 available.

At capture time the thermal service reported overall status 0, while individual
sensor values included battery 39.9 C, SoC 56.707 C, GPU 49.33 C, and power
amplifier 43.419 C. These values only prove that thermal telemetry is
discoverable. They are not benchmark measurements and must not be reused as a
trial baseline.

## Immediate implications

- The connected phone can support the first feasibility pilot and thermal
  metadata collection.
- Full-backbone feasibility passed for all three architectures at batch 8;
  batch 16 is unsuitable because EfficientNet-B0 exhausted GPU memory.
- Confirmatory claims about hardware-aware selection need more device diversity
  than the one phone currently connected.
- The Java/JDK path, NNAPI support, delegate availability, and per-delegate
  execution must be tested rather than inferred.
