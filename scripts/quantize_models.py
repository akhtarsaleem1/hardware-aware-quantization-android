"""CLI for traceable TFLite conversion using an NPZ calibration tensor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantization.tflite_variants import convert_keras_variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/tflite")
    parser.add_argument(
        "--representative-npz",
        type=Path,
        help="NPZ created from training data only; must contain an 'inputs' array.",
    )
    parser.add_argument("--model-id")
    args = parser.parse_args()

    representative = None
    if args.representative_npz:
        calibration_path = args.representative_npz.resolve()

        def representative():
            with np.load(calibration_path) as archive:
                if "inputs" not in archive:
                    raise KeyError("Representative NPZ must contain 'inputs'")
                for sample in archive["inputs"]:
                    yield sample[None, ...]

    status = convert_keras_variants(
        args.model, args.output_dir, representative, args.model_id
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

