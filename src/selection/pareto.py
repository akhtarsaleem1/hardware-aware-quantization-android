"""Pareto-front selection without arbitrary utility weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def pareto_efficient_indices(
    records: Sequence[Mapping[str, Any]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> list[int]:
    """Return indices not dominated by another complete record.

    A record dominates another when it is at least as good in every objective
    and strictly better in at least one. Missing and non-finite objectives are
    rejected because silently imputing them could change the selected front.
    """

    objectives = list(maximize) + list(minimize)
    if not objectives:
        raise ValueError("At least one objective is required")
    if len(set(objectives)) != len(objectives):
        raise ValueError("An objective cannot be both maximized and minimized")
    if not records:
        return []

    rows: list[list[float]] = []
    for index, record in enumerate(records):
        try:
            row = [float(record[name]) for name in objectives]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Record {index} has incomplete objectives") from exc
        if not np.all(np.isfinite(row)):
            raise ValueError(f"Record {index} has non-finite objectives")
        rows.append(row)

    values = np.asarray(rows, dtype=np.float64)
    if minimize:
        values[:, len(maximize) :] *= -1.0
    efficient: list[int] = []
    for candidate in range(len(records)):
        at_least_as_good = np.all(values >= values[candidate], axis=1)
        strictly_better = np.any(values > values[candidate], axis=1)
        dominated = np.any(at_least_as_good & strictly_better)
        if not dominated:
            efficient.append(candidate)
    return efficient

