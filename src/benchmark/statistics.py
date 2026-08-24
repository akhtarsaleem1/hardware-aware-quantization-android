"""Statistics derived from immutable per-inference latency observations."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def summarize_latencies(latencies_ms: Iterable[float]) -> dict[str, Any]:
    """Return a documented descriptive summary in milliseconds.

    The confidence interval uses a normal approximation around the arithmetic
    mean. Confirmatory analysis may replace it with a preregistered hierarchical
    or bootstrap interval across complete trials.
    """

    values = np.asarray(list(latencies_ms), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("At least one one-dimensional latency value is required")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Latency values must be finite and greater than zero")
    count = int(values.size)
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if count > 1 else 0.0
    margin = 1.96 * standard_deviation / np.sqrt(count) if count > 1 else 0.0
    return {
        "count": count,
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "mean_ms": mean,
        "median_ms": float(np.median(values)),
        "standard_deviation_ms": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean,
        "mean_ci95_low_ms": mean - margin,
        "mean_ci95_high_ms": mean + margin,
        "mean_ci95_method": "normal_approximation_1.96",
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "sequential_throughput_per_second_from_median": 1000.0
        / float(np.median(values)),
    }

