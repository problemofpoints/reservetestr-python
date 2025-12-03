"""Reservetestr Python: stochastic reserving backtesting helpers."""

from .datasets import (
    TriangleRecord,
    load_clrd_dataframe,
    load_meyers_subset,
    build_triangle_records,
)
from .backtesting import run_single_backtest
from .methods import (
    testr_mack_chainladder,
    testr_bootstrap_odp,
)
from .plots import create_pp_plot, create_histogram_plot
from .framework import (
    create_test_method,
    create_bootstrap_test_method,
    make_test_method,
    get_triangle,
    resolve_actual_ultimate,
    calculate_derived_metrics,
)

__all__ = [
    # Dataset loaders
    "TriangleRecord",
    "load_clrd_dataframe",
    "load_meyers_subset",
    "build_triangle_records",
    # Backtesting
    "run_single_backtest",
    # Pre-built test methods
    "testr_mack_chainladder",
    "testr_bootstrap_odp",
    # Plotting
    "create_pp_plot",
    "create_histogram_plot",
    # Framework for creating custom test methods
    "create_test_method",
    "create_bootstrap_test_method",
    "make_test_method",
    "get_triangle",
    "resolve_actual_ultimate",
    "calculate_derived_metrics",
]
