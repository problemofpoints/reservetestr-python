"""Flexible testing framework for reserving methods.

This module provides the plumbing to create test functions for any reserving method
that can produce ultimate estimates and standard deviations. Users don't need to create
method-specific test functions - they can use the generic framework.

Basic usage:
    >>> from reservetestr.framework import create_test_method
    >>> import chainladder as cl
    >>>
    >>> def fit_my_method(triangle, **kwargs):
    ...     model = cl.MackChainladder(**kwargs).fit(triangle)
    ...     return model
    >>>
    >>> def extract_estimates(model, triangle):
    ...     mean_ult = float(model.ultimate_.sum().values)
    ...     stddev = float(model.total_mack_std_err_.values.squeeze())
    ...     return mean_ult, stddev
    >>>
    >>> test_my_method = create_test_method(
    ...     fit_func=fit_my_method,
    ...     extract_func=extract_estimates
    ... )
    >>>
    >>> # Use with run_single_backtest
    >>> results = rt.run_single_backtest(
    ...     records, test_my_method, method_label="my_method"
    ... )

Alternative: For even simpler cases, use make_test_method decorator:
    >>> @make_test_method
    ... def test_simple_method(triangle, **kwargs):
    ...     model = SomeModel(**kwargs).fit(triangle)
    ...     mean_ult = extract_ultimate(model)
    ...     stddev = extract_stddev(model)
    ...     return mean_ult, stddev
"""
from __future__ import annotations

from functools import wraps
from typing import Callable, Dict, Optional, Protocol, Tuple

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


class ReservingModel(Protocol):
    """Protocol for a fitted reserving model.

    A model fitting function should accept a triangle and return an object
    with this interface (or return values that the extract function knows how to handle).
    """
    pass


def get_triangle(
    triangles: LossTypeMapping, loss_type: str
) -> Optional[cl.Triangle]:
    """Extract a triangle of the specified loss type.

    Args:
        triangles: Dictionary mapping loss types to Triangle objects
        loss_type: The loss type to extract (e.g., 'paid', 'incurred')

    Returns:
        The requested Triangle or None if not available

    Raises:
        ValueError: If loss_type is not a valid key
    """
    if loss_type not in triangles:
        raise ValueError(f"Unknown loss_type '{loss_type}'")
    return triangles[loss_type]


def resolve_actual_ultimate(
    actual_ultimates: Optional[Dict[str, float]],
    test_triangle: Optional[cl.Triangle],
    loss_type: str,
    train_triangle: cl.Triangle,
) -> Optional[float]:
    """Determine the actual ultimate value for comparison.

    Priority order:
    1. Use actual_ultimates[loss_type] if available and not NaN
    2. Use latest diagonal sum from test_triangle if available
    3. Return None if neither is available

    Args:
        actual_ultimates: Dictionary of known ultimate values by loss type
        test_triangle: Holdout triangle (if available)
        loss_type: The loss type being evaluated
        train_triangle: Training triangle (used if test unavailable)

    Returns:
        Actual ultimate value or None
    """
    # Try actual_ultimates first
    if actual_ultimates:
        value = actual_ultimates.get(loss_type)
        if value is not None and not np.isnan(value):
            return float(value)

    # Fall back to test triangle
    if test_triangle is not None:
        return latest_cumulative_sum(test_triangle)

    return None


def calculate_derived_metrics(
    mean_ultimate_est: float,
    stddev_est: float,
    actual_ultimate: float,
    latest_observed: float,
    distribution: str = "lognormal",
    samples: Optional[np.ndarray] = None,
) -> dict:
    """Calculate standard derived metrics from estimates and actuals.

    Args:
        mean_ultimate_est: Estimated ultimate loss
        stddev_est: Standard deviation of ultimate estimate
        actual_ultimate: Observed actual ultimate
        latest_observed: Latest observed cumulative loss
        distribution: Distribution to use for implied percentile calculation
                     ('lognormal', 'normal', or 'empirical')
        samples: Array of simulated samples (required if distribution='empirical')

    Returns:
        Dictionary with keys: actual_unpaid, mean_unpaid_est, cv_unpaid_est, implied_pctl
    """
    actual_unpaid = actual_ultimate - latest_observed
    mean_unpaid_est = mean_ultimate_est - latest_observed
    cv_unpaid_est = safe_divide(stddev_est, mean_unpaid_est)

    # Calculate implied percentile based on distribution
    if distribution == "lognormal":
        implied_pctl = calculate_lognormal_percentile(
            actual_ultimate, mean_ultimate_est, stddev_est
        )
    elif distribution == "normal":
        implied_pctl = calculate_normal_percentile(
            actual_ultimate, mean_ultimate_est, stddev_est
        )
    elif distribution == "empirical":
        if samples is None:
            raise ValueError("samples required for empirical distribution")
        implied_pctl = calculate_empirical_percentile(actual_ultimate, samples)
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Must be 'lognormal', 'normal', or 'empirical'"
        )

    return {
        "actual_unpaid": actual_unpaid,
        "mean_unpaid_est": mean_unpaid_est,
        "cv_unpaid_est": cv_unpaid_est,
        "implied_pctl": implied_pctl,
    }


def calculate_lognormal_percentile(
    actual: float, mean_est: float, stddev_est: float
) -> float:
    """Calculate implied percentile assuming lognormal distribution.

    Args:
        actual: Actual observed value
        mean_est: Mean of the estimate
        stddev_est: Standard deviation of the estimate

    Returns:
        Implied percentile (0-1) or NaN if calculation fails
    """
    cv_total = safe_divide(stddev_est, mean_est)
    try:
        params = moments_to_lognormal(mean_est, cv_total)
        return float(
            lognorm.cdf(
                actual,
                s=params.sdlog,
                scale=np.exp(params.meanlog),
            )
        )
    except (ValueError, FloatingPointError):
        return float("nan")


def calculate_normal_percentile(
    actual: float, mean_est: float, stddev_est: float
) -> float:
    """Calculate implied percentile assuming normal distribution.

    Args:
        actual: Actual observed value
        mean_est: Mean of the estimate
        stddev_est: Standard deviation of the estimate

    Returns:
        Implied percentile (0-1) or NaN if calculation fails
    """
    from scipy.stats import norm

    if not np.isfinite(stddev_est) or stddev_est <= 0:
        return float("nan")
    try:
        return float(norm.cdf(actual, loc=mean_est, scale=stddev_est))
    except (ValueError, FloatingPointError):
        return float("nan")


def calculate_empirical_percentile(actual: float, samples: np.ndarray) -> float:
    """Calculate implied percentile from empirical samples.

    Args:
        actual: Actual observed value
        samples: Array of simulated values

    Returns:
        Implied percentile (0-1) or NaN if calculation fails
    """
    if not np.isfinite(actual):
        return float("nan")
    return float(np.nanmean(samples <= actual))


def create_test_method(
    fit_func: Callable[[cl.Triangle, ...], ReservingModel],
    extract_func: Callable[[ReservingModel, cl.Triangle], Tuple[float, float]],
    distribution: str = "lognormal",
    percentile_func: Optional[Callable[[float, ReservingModel, cl.Triangle], float]] = None,
) -> Callable:
    """Create a test method wrapper for any reserving method.

    This factory function creates a standardized test function that can be used
    with run_single_backtest(). You only need to provide:
    1. How to fit your model
    2. How to extract estimates from the fitted model

    Args:
        fit_func: Function that fits a model to a triangle.
                 Signature: fit_func(triangle, **kwargs) -> model
        extract_func: Function that extracts estimates from a fitted model.
                     Signature: extract_func(model, triangle) -> (mean_ultimate, stddev)
        distribution: Distribution to assume for implied percentile calculation.
                     Options: 'lognormal', 'normal', 'empirical'
        percentile_func: Optional custom function to calculate implied percentile.
                        Signature: percentile_func(actual, model, triangle) -> float
                        If provided, overrides the distribution parameter.

    Returns:
        A test function compatible with run_single_backtest()

    Example:
        >>> def fit_my_model(triangle, alpha=1.0):
        ...     return MyModel(alpha=alpha).fit(triangle)
        >>>
        >>> def extract_my_estimates(model, triangle):
        ...     mean = float(model.ultimate_.sum().values)
        ...     std = float(model.std_err_.sum().values)
        ...     return mean, std
        >>>
        >>> test_my_model = create_test_method(fit_my_model, extract_my_estimates)
        >>>
        >>> results = run_single_backtest(
        ...     records, test_my_model, alpha=1.5, method_label="my_method"
        ... )
    """

    def test_method(
        train_triangles: LossTypeMapping,
        test_triangles: LossTypeMapping,
        loss_type: str = "paid",
        actual_ultimates: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Optional[dict]:
        """Test method created by create_test_method."""

        # Extract triangles
        triangle = get_triangle(train_triangles, loss_type)
        test_triangle = get_triangle(test_triangles, loss_type)

        if triangle is None:
            return None

        # Fit model
        model = fit_func(triangle, **kwargs)

        # Extract estimates
        mean_ultimate_est, stddev_est = extract_func(model, triangle)

        # Resolve actual ultimate
        actual_ultimate = resolve_actual_ultimate(
            actual_ultimates, test_triangle, loss_type, triangle
        )
        if actual_ultimate is None:
            return None

        # Calculate latest observed
        latest_observed = latest_cumulative_sum(triangle)

        # Calculate derived metrics
        if percentile_func is not None:
            # Use custom percentile function
            implied_pctl = percentile_func(actual_ultimate, model, triangle)
            # Calculate other metrics manually
            actual_unpaid = actual_ultimate - latest_observed
            mean_unpaid_est = mean_ultimate_est - latest_observed
            cv_unpaid_est = safe_divide(stddev_est, mean_unpaid_est)
            derived = {
                "actual_unpaid": actual_unpaid,
                "mean_unpaid_est": mean_unpaid_est,
                "cv_unpaid_est": cv_unpaid_est,
                "implied_pctl": implied_pctl,
            }
        else:
            # Use standard distribution-based calculation
            derived = calculate_derived_metrics(
                mean_ultimate_est,
                stddev_est,
                actual_ultimate,
                latest_observed,
                distribution=distribution,
            )

        return {
            "actual_ultimate": actual_ultimate,
            "mean_ultimate_est": mean_ultimate_est,
            "stddev_est": stddev_est,
            **derived,
        }

    return test_method


def make_test_method(
    distribution: str = "lognormal",
) -> Callable:
    """Decorator to convert a simple fit function into a test method.

    Use this decorator when your function fits a model and returns (mean, stddev).
    The decorator handles all the boilerplate of triangle extraction, actual
    ultimate calculation, and derived metrics.

    Args:
        distribution: Distribution to assume for implied percentile calculation.
                     Options: 'lognormal', 'normal'

    Returns:
        Decorator function

    Example:
        >>> @make_test_method(distribution='lognormal')
        ... def test_mack(triangle, **kwargs):
        ...     model = cl.MackChainladder(**kwargs).fit(triangle)
        ...     mean_ult = float(model.ultimate_.sum().values)
        ...     stddev = float(model.total_mack_std_err_.values.squeeze())
        ...     return mean_ult, stddev
        >>>
        >>> results = run_single_backtest(records, test_mack, method_label="mack")
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def test_method(
            train_triangles: LossTypeMapping,
            test_triangles: LossTypeMapping,
            loss_type: str = "paid",
            actual_ultimates: Optional[Dict[str, float]] = None,
            **kwargs,
        ) -> Optional[dict]:
            """Test method created by make_test_method decorator."""

            # Extract triangles
            triangle = get_triangle(train_triangles, loss_type)
            test_triangle = get_triangle(test_triangles, loss_type)

            if triangle is None:
                return None

            # Call user function to get estimates
            mean_ultimate_est, stddev_est = func(triangle, **kwargs)

            # Resolve actual ultimate
            actual_ultimate = resolve_actual_ultimate(
                actual_ultimates, test_triangle, loss_type, triangle
            )
            if actual_ultimate is None:
                return None

            # Calculate latest observed
            latest_observed = latest_cumulative_sum(triangle)

            # Calculate derived metrics
            derived = calculate_derived_metrics(
                mean_ultimate_est,
                stddev_est,
                actual_ultimate,
                latest_observed,
                distribution=distribution,
            )

            return {
                "actual_ultimate": actual_ultimate,
                "mean_ultimate_est": mean_ultimate_est,
                "stddev_est": stddev_est,
                **derived,
            }

        return test_method

    return decorator


def create_bootstrap_test_method(
    sample_func: Callable[[cl.Triangle, ...], np.ndarray],
    bootstrap_class: Optional[type] = None,
    **default_bootstrap_kwargs,
) -> Callable:
    """Create a test method for bootstrap/simulation-based methods.

    This factory is specialized for methods that generate samples of ultimates
    (like bootstrap or MCMC methods). It handles the empirical percentile calculation.

    Args:
        sample_func: Function that generates samples from a triangle.
                    Signature: sample_func(triangle, **kwargs) -> np.ndarray of samples
        bootstrap_class: Optional chainladder bootstrap class (e.g., cl.BootstrapODPSample)
        **default_bootstrap_kwargs: Default kwargs for the bootstrap class

    Returns:
        A test function compatible with run_single_backtest()

    Example:
        >>> def generate_samples(triangle, n_sims=1000, **kwargs):
        ...     bootstrap = cl.BootstrapODPSample(n_sims=n_sims, **kwargs).fit(triangle)
        ...     resampled = bootstrap.transform(triangle)
        ...     model = cl.Chainladder().fit(resampled)
        ...     return model.ultimate_.sum(axis=(1,2,3)).values.flatten()
        >>>
        >>> test_bootstrap = create_bootstrap_test_method(generate_samples)
    """

    def test_method(
        train_triangles: LossTypeMapping,
        test_triangles: LossTypeMapping,
        loss_type: str = "paid",
        actual_ultimates: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Optional[dict]:
        """Test method for bootstrap/simulation methods."""

        # Extract triangles
        triangle = get_triangle(train_triangles, loss_type)
        test_triangle = get_triangle(test_triangles, loss_type)

        if triangle is None:
            return None

        # Generate samples
        samples = sample_func(triangle, **kwargs)

        if samples.size == 0:
            return None

        # Calculate estimates from samples
        mean_ultimate_est = float(np.nanmean(samples))
        stddev_est = (
            float(np.nanstd(samples, ddof=1)) if samples.size > 1 else float("nan")
        )

        # Resolve actual ultimate
        actual_ultimate = resolve_actual_ultimate(
            actual_ultimates, test_triangle, loss_type, triangle
        )
        if actual_ultimate is None:
            return None

        # Calculate latest observed
        latest_observed = latest_cumulative_sum(triangle)

        # Calculate derived metrics using empirical distribution
        derived = calculate_derived_metrics(
            mean_ultimate_est,
            stddev_est,
            actual_ultimate,
            latest_observed,
            distribution="empirical",
            samples=samples,
        )

        return {
            "actual_ultimate": actual_ultimate,
            "mean_ultimate_est": mean_ultimate_est,
            "stddev_est": stddev_est,
            **derived,
        }

    return test_method
