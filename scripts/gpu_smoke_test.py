"""Verify that TensorFlow executes forward and backward tensor work on a GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--matrix-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise SystemExit("GPU_REQUIRED: TensorFlow reports no physical GPU")
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
    with tf.device("/GPU:0"):
        left = tf.Variable(tf.random.normal((args.matrix_size, args.matrix_size)))
        right = tf.random.normal((args.matrix_size, args.matrix_size))
        with tf.GradientTape() as tape:
            result = tf.reduce_sum(tf.matmul(left, right))
        gradient = tape.gradient(result, left)
    if gradient is None:
        raise SystemExit("GPU_REQUIRED: gradient was not produced")
    if "GPU:" not in result.device.upper() or "GPU:" not in gradient.device.upper():
        raise SystemExit(
            f"GPU_REQUIRED: unexpected devices result={result.device} gradient={gradient.device}"
        )
    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        nvidia_smi = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "physical_gpus": [device.name for device in physical],
        "nvidia_smi": nvidia_smi,
        "matrix_size": args.matrix_size,
        "result_device": result.device,
        "gradient_device": gradient.device,
        "memory_info": tf.config.experimental.get_memory_info("GPU:0"),
        "status": "PASS",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
