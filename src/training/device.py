"""Fail-closed GPU selection and environment capture."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess


def require_gpu(tf, report_path: str | Path) -> dict:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError(
            "GPU_REQUIRED: TensorFlow reports no physical GPU; refusing CPU training"
        )
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
    try:
        smi = subprocess.run(
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
        smi = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    details = []
    for device in physical:
        item = {"name": device.name, "device_type": device.device_type}
        item.update(tf.config.experimental.get_device_details(device))
        details.append(item)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "physical_gpus": details,
        "nvidia_smi": smi,
        "policy": "GPU_REQUIRED_ABORT_ON_CPU_FALLBACK",
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def assert_tensor_on_gpu(tensor) -> None:
    if "GPU:" not in tensor.device.upper():
        raise RuntimeError(f"GPU_REQUIRED: smoke tensor was placed on {tensor.device}")
