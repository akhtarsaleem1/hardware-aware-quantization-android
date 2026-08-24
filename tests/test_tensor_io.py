from __future__ import annotations

import unittest

import numpy as np

from src.quantization.tensor_io import dequantize_output, quantize_input


class TensorIoTests(unittest.TestCase):
    def test_float_input_preserves_values_in_requested_dtype(self) -> None:
        values = np.array([-1.0, 0.5, 2.0], dtype=np.float64)
        result = quantize_input(values, {"dtype": np.float32})
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, values)

    def test_int8_round_trip_respects_scale_and_zero_point(self) -> None:
        detail = {"dtype": np.int8, "quantization": (0.25, -3)}
        values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        quantized = quantize_input(values, detail)
        np.testing.assert_array_equal(quantized, np.array([-7, -3, 1], dtype=np.int8))
        np.testing.assert_allclose(
            dequantize_output(quantized, detail), values, atol=0.125
        )

    def test_integer_input_clips_instead_of_wrapping(self) -> None:
        detail = {"dtype": np.int8, "quantization": (0.1, 0)}
        result = quantize_input(np.array([-1000.0, 1000.0]), detail)
        np.testing.assert_array_equal(result, np.array([-128, 127], dtype=np.int8))

    def test_invalid_integer_scale_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quantize_input(
                np.array([1.0]), {"dtype": np.int8, "quantization": (0.0, 0)}
            )


if __name__ == "__main__":
    unittest.main()

