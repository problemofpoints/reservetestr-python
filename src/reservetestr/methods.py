"""Wrappers around chainladder reserving methods."""
from __future__ import annotations

from typing import Dict, Optional

import chainladder as cl
import numpy as np
from scipy.stats import lognorm

from .utils import (
    latest_cumulative_sum,
    moments_to_lognormal,
    safe_divide,
    triangle_total,
)

LossTypeMapping = Dict[str, Optional[cl.Triangle]]


def _resolve_actual_ultimate(
    actual_ultimates: Optional[Dict[str, float]], loss_type: str
) -> Optional[float]:
    if not actual_ultimates:
        return None
    value = actual_ultimates.get(loss_type)
    if value is None or np.isnan(value):
        return None
    return float(value)


def _get_triangle(triangles: LossTypeMapping, loss_type: str) -> Optional[cl.Triangle]:
    if loss_type not in triangles:
        raise ValueError(f"Unknown loss_type '{loss_type}'")
    return triangles[loss_type]


def testr_mack_chainladder(
    train_triangles: LossTypeMapping,
    test_triangles: LossTypeMapping,
    loss_type: str = "paid",
    actual_ultimates: Optional[Dict[str, float]] = None,
    model_kwargs: Optional[dict] = None,
) -> Optional[dict]:
    """Mirror the reservetestr::testr_MackChainLadder helper."""

    triangle = _get_triangle(train_triangles, loss_type)
    test_triangle = _get_triangle(test_triangles, loss_type)
    if triangle is None:
        return None

    model = cl.MackChainladder(**(model_kwargs or {})).fit(triangle)
    method_ultimate = triangle_total(model.ultimate_)
    stddev_df = model.total_mack_std_err_
    stddev_est = float(np.asarray(stddev_df.values).squeeze())

    latest_observed = latest_cumulative_sum(triangle)
    actual_ultimate = _resolve_actual_ultimate(actual_ultimates, loss_type)
    if actual_ultimate is None:
        if test_triangle is None:
            return None
        actual_ultimate = latest_cumulative_sum(test_triangle)
    actual_unpaid = actual_ultimate - latest_observed
    mean_unpaid_est = method_ultimate - latest_observed
    cv_total = safe_divide(stddev_est, method_ultimate)

    try:
        params = moments_to_lognormal(method_ultimate, cv_total)
        implied_pctl = float(
            lognorm.cdf(
                actual_ultimate,
                s=params.sdlog,
                scale=np.exp(params.meanlog),
            )
        )
    except (ValueError, FloatingPointError):
        implied_pctl = np.nan

    cv_unpaid_est = safe_divide(stddev_est, mean_unpaid_est)

    return {
        "actual_ultimate": actual_ultimate,
        "actual_unpaid": actual_unpaid,
        "mean_ultimate_est": method_ultimate,
        "mean_unpaid_est": mean_unpaid_est,
        "stddev_est": stddev_est,
        "cv_unpaid_est": cv_unpaid_est,
        "implied_pctl": implied_pctl,
    }


def _bootstrap_samples(triangle: cl.Triangle) -> np.ndarray:
    """Fit Chainladder to each simulated triangle and pull total ultimates."""
    model = cl.Chainladder().fit(triangle)
    values = np.asarray(model.ultimate_.values, dtype=float)
    return np.nansum(values, axis=(1, 2, 3))


def testr_bootstrap_odp(
    train_triangles: LossTypeMapping,
    test_triangles: LossTypeMapping,
    loss_type: str = "paid",
    actual_ultimates: Optional[Dict[str, float]] = None,
    n_sims: int = 1000,
    hat_adj: bool = False,
    random_state: Optional[int] = None,
    **bootstrap_kwargs,
) -> Optional[dict]:
    """Mirror the reservetestr::testr_BootChainLadder helper using BootstrapODPSample."""

    triangle = _get_triangle(train_triangles, loss_type)
    test_triangle = _get_triangle(test_triangles, loss_type)
    if triangle is None:
        return None

    bootstrap = cl.bootstrap.BootstrapODPSample(
        n_sims=n_sims,
        hat_adj=hat_adj,
        random_state=random_state,
        **bootstrap_kwargs,
    ).fit(triangle)
    resampled = bootstrap.transform(triangle)
    samples = _bootstrap_samples(resampled)
    if samples.size == 0:
        return None

    mean_ultimate = float(np.nanmean(samples))
    stddev_est = float(np.nanstd(samples, ddof=1)) if samples.size > 1 else float("nan")
    latest_observed = latest_cumulative_sum(triangle)
    actual_ultimate = _resolve_actual_ultimate(actual_ultimates, loss_type)
    if actual_ultimate is None:
        if test_triangle is None:
            return None
        actual_ultimate = latest_cumulative_sum(test_triangle)
    actual_unpaid = actual_ultimate - latest_observed
    mean_unpaid_est = mean_ultimate - latest_observed
    cv_unpaid_est = safe_divide(stddev_est, mean_unpaid_est)
    implied_pctl = float(np.nanmean(samples <= actual_ultimate)) if np.isfinite(actual_ultimate) else np.nan

    return {
        "actual_ultimate": actual_ultimate,
        "actual_unpaid": actual_unpaid,
        "mean_ultimate_est": mean_ultimate,
        "mean_unpaid_est": mean_unpaid_est,
        "stddev_est": stddev_est,
        "cv_unpaid_est": cv_unpaid_est,
        "implied_pctl": implied_pctl,
    }
