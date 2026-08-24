"""Verify existing TFLite flatbuffers without automatic desktop delegates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_metadata(detail: dict) -> dict:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return {
        "name": str(detail.get("name", "")),
        "shape": [int(value) for value in detail["shape"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "scale": float(scale),
        "zero_point": int(zero_point),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import tensorflow as tf

    variants = {}
    for path in sorted(args.directory.glob("*.tflite")):
        try:
            interpreter = tf.lite.Interpreter(
                model_path=str(path),
                experimental_op_resolver_type=(
                    tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                ),
            )
            interpreter.allocate_tensors()
            variants[path.stem] = {
                "status": "success",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "inputs": [tensor_metadata(item) for item in interpreter.get_input_details()],
                "outputs": [tensor_metadata(item) for item in interpreter.get_output_details()],
            }
        except Exception as exc:
            variants[path.stem] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    report = {
        "status": "PASS" if variants and all(v["status"] == "success" for v in variants.values()) else "FAIL",
        "purpose": "flatbuffer allocation only; delegate compatibility is separate",
        "runtime": "builtin_without_default_delegates",
        "variants": variants,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
