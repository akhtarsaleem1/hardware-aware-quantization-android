"""Create traceable FP32, FP16, dynamic-range, and full-INT8 TFLite files."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


RepresentativeFactory = Callable[[], Iterable[np.ndarray | list[np.ndarray]]]


def _representative_generator(factory: RepresentativeFactory) -> Iterator[list[np.ndarray]]:
    for sample in factory():
        tensors = sample if isinstance(sample, list) else [sample]
        yield [np.asarray(tensor, dtype=np.float32) for tensor in tensors]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_metadata(detail: dict[str, Any]) -> dict[str, Any]:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return {
        "name": str(detail.get("name", "")),
        "shape": [int(value) for value in detail["shape"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "scale": float(scale),
        "zero_point": int(zero_point),
    }


def convert_keras_variants(
    keras_model_path: Path,
    output_dir: Path,
    representative_factory: RepresentativeFactory | None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Attempt all planned variants and retain every success or failure.

    TensorFlow is imported lazily so metadata/statistics tooling remains usable
    on machines that do not train or convert models.
    """

    import tensorflow as tf

    keras_model_path = keras_model_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(keras_model_path)
    artifact_id = model_id or keras_model_path.stem

    def base() -> Any:
        return tf.lite.TFLiteConverter.from_keras_model(model)

    def dynamic() -> Any:
        converter = base()
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter

    def float16() -> Any:
        converter = dynamic()
        converter.target_spec.supported_types = [tf.float16]
        return converter

    def full_int8() -> Any:
        if representative_factory is None:
            raise ValueError("Full INT8 requires representative training data")
        converter = dynamic()
        converter.representative_dataset = lambda: _representative_generator(
            representative_factory
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        return converter

    variants: dict[str, Callable[[], Any]] = {
        "float32": base,
        "float16": float16,
        "dynamic_int8": dynamic,
        "full_int8": full_int8,
    }
    status: dict[str, Any] = {
        "source_model": str(keras_model_path),
        "source_sha256": _sha256(keras_model_path),
        "representative_data_supplied": representative_factory is not None,
        "variants": {},
    }
    for variant, factory in variants.items():
        destination = output_dir / f"{artifact_id}__{variant}.tflite"
        try:
            destination.write_bytes(factory().convert())
            interpreter = tf.lite.Interpreter(model_path=str(destination))
            interpreter.allocate_tensors()
            status["variants"][variant] = {
                "status": "success",
                "path": str(destination.resolve()),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "inputs": [
                    _tensor_metadata(value) for value in interpreter.get_input_details()
                ],
                "outputs": [
                    _tensor_metadata(value) for value in interpreter.get_output_details()
                ],
            }
        except Exception as exc:  # Runtime/version-specific failures are evidence.
            status["variants"][variant] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    status_path = output_dir / f"{artifact_id}__conversion_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status

