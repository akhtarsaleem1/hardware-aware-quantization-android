"""Numerically explicit TensorFlow Lite input/output conversion helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _quantization_parameters(detail: Mapping[str, Any]) -> tuple[float, int]:
    """Return per-tensor scale and zero point or raise a useful error.

    The initial study intentionally supports per-tensor model input/output
    quantization. Per-axis activation tensors must be detected and implemented
    explicitly rather than being silently reduced to one scale.
    """

    scale, zero_point = detail.get("quantization", (0.0, 0))
    scale = float(scale)
    zero_point = int(zero_point)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Integer tensor has no valid positive quantization scale")
    return scale, zero_point


def quantize_input(values: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    """Convert float preprocessing output to the interpreter input dtype."""

    dtype = np.dtype(detail["dtype"])
    array = np.asarray(values)
    if np.issubdtype(dtype, np.floating):
        return array.astype(dtype, copy=False)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"Unsupported TensorFlow Lite input dtype: {dtype}")
    scale, zero_point = _quantization_parameters(detail)
    limits = np.iinfo(dtype)
    quantized = np.rint(array.astype(np.float64) / scale + zero_point)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def dequantize_output(values: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    """Convert interpreter output to float32 for fair metric computation."""

    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32, copy=False)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"Unsupported TensorFlow Lite output dtype: {array.dtype}")
    scale, zero_point = _quantization_parameters(detail)
    return (array.astype(np.float32) - zero_point) * scale

