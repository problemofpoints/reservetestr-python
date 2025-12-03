"""Wrappers around chainladder reserving methods.

This module demonstrates how to use the flexible testing framework to create
test methods for different reserving approaches. The examples show both:
1. Using create_test_method() for analytical methods
2. Using create_bootstrap_test_method() for simulation-based methods
3. Direct implementation for full control

All of these approaches work with run_single_backtest().
"""
from __future__ import annotations

from typing import Dict, Optional

import chainladder as cl
import numpy as np

from .framework import create_bootstrap_test_method, create_test_method
from .utils import triangle_total

LossTypeMapping = Dict[str, Optional[cl.Triangle]]


# ==============================================================================
# Example 1: Mack Chain Ladder using create_test_method()
# ==============================================================================

def _fit_mack_chainladder(triangle: cl.Triangle, model_kwargs: Optional[dict] = None, **kwargs) -> cl.MackChainladder:
    """Fit a Mack Chain Ladder model to a triangle."""
    return cl.MackChainladder(**(model_kwargs or {})).fit(triangle)


def _extract_mack_estimates(model: cl.MackChainladder, triangle: cl.Triangle) -> tuple[float, float]:
    """Extract ultimate and stddev from a fitted Mack model."""
    mean_ultimate = triangle_total(model.ultimate_)
    stddev = float(np.asarray(model.total_mack_std_err_.values).squeeze())
    return mean_ultimate, stddev


# Create the test method using the framework
testr_mack_chainladder = create_test_method(
    fit_func=_fit_mack_chainladder,
    extract_func=_extract_mack_estimates,
    distribution="lognormal",
)


# ==============================================================================
# Example 2: Bootstrap ODP using create_bootstrap_test_method()
# ==============================================================================

def _generate_bootstrap_samples(
    triangle: cl.Triangle,
    n_sims: int = 1000,
    hat_adj: bool = False,
    random_state: Optional[int] = None,
    **bootstrap_kwargs,
) -> np.ndarray:
    """Generate bootstrap samples using BootstrapODPSample."""
    bootstrap = cl.BootstrapODPSample(
        n_sims=n_sims,
        hat_adj=hat_adj,
        random_state=random_state,
        **bootstrap_kwargs,
    ).fit(triangle)
    resampled = bootstrap.transform(triangle)

    # Fit Chainladder to each simulated triangle and extract ultimates
    model = cl.Chainladder().fit(resampled)
    values = np.asarray(model.ultimate_.values, dtype=float)
    return np.nansum(values, axis=(1, 2, 3))


# Create the test method using the bootstrap framework
testr_bootstrap_odp = create_bootstrap_test_method(
    sample_func=_generate_bootstrap_samples,
)
