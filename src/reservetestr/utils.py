"""Utility helpers for reservetestr."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LognormalParams:
    meanlog: float
    sdlog: float


def moments_to_lognormal(mean: float, cv: float) -> LognormalParams:
    """Convert mean/cv to lognormal parameters."""
    if mean <= 0:
        raise ValueError("mean must be positive for lognormal conversion")
    if cv <= 0:
        raise ValueError("cv must be positive for lognormal conversion")
    sdlog = np.sqrt(np.log1p(cv ** 2))
    meanlog = np.log(mean) - 0.5 * sdlog ** 2
    return LognormalParams(meanlog=meanlog, sdlog=sdlog)


def latest_cumulative_sum(triangle) -> float:
    """Return the sum of the latest cumulative values for a Triangle."""
    latest = triangle.latest_diagonal
    total = np.nansum(latest.values)
    return float(total)


def triangle_total(triangle) -> float:
    """Sum all numeric values in a Triangle."""
    return float(np.nansum(triangle.values))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def ensure_iterable(obj) -> Iterable:
    if isinstance(obj, (list, tuple)):
        return obj
    return (obj,)
