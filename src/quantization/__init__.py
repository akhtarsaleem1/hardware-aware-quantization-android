"""TensorFlow Lite quantization and tensor-I/O helpers."""

from .tensor_io import dequantize_output, quantize_input

__all__ = ["dequantize_output", "quantize_input"]

