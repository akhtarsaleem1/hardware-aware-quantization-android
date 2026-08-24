# Compute policy: GPU-first for heavy work

The workstation has an NVIDIA GeForce GTX 1650 with 4 GiB of VRAM. Model
training, large evaluation passes, and other genuinely heavy tensor workloads
must use the GPU when the selected framework and operation support it.

## Required behavior

1. Before a heavy run, print and save framework version, visible accelerators,
   selected device, GPU name, and available VRAM.
2. Abort a GPU-required experiment if the framework selects CPU unexpectedly.
   Do not silently continue and later describe the run as GPU-accelerated.
3. Use mixed precision only after a numerical smoke test and only when supported
   by the model/GPU path.
4. Reduce batch size or use gradient accumulation when 4 GiB VRAM is
   insufficient; do not move the full training job to CPU merely to avoid OOM.
5. CPU is appropriate for dataset indexing, CSV/statistical processing, unit
   tests, TFLite CPU benchmarking, and small conversion-control tasks.
6. Android inference experiments must use the requested on-device runtime; the
   laptop GPU is irrelevant to phone latency evidence.

## Verified platform

Native-Windows TensorFlow GPU compatibility is not assumed. The selected path
is TensorFlow under WSL2, preserving a direct Keras-to-LiteRT conversion path.
The verified environment is:

- Ubuntu 26.04 under WSL2, kernel 6.18.33.2;
- uv-managed CPython 3.13.15;
- TensorFlow 2.21.0 with NVIDIA CUDA wheel libraries;
- NVIDIA GeForce GTX 1650, compute capability 7.5;
- 2,244 MiB made available to the TensorFlow process.

The matrix-and-gradient smoke test and all full-backbone feasibility steps ran
on `/GPU:0`. Batch 16 was rejected after EfficientNet-B0 produced a GPU OOM;
batch 8 passed for every frozen architecture and is now the common training
batch size. The failed run is retained rather than hidden.

The NVIDIA libraries installed inside the Python environment must be present in
the process library search path before TensorFlow starts. Use
`scripts/run_wsl_gpu_training.sh`; the training script contains a second guard
that exits before loading the dataset unless TensorFlow reports a physical GPU.
