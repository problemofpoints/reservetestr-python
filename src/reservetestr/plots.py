"""Visualization helpers mirroring the original R package exhibits."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_REQUIRED_COLS = {
    "line",
    "group_id",
    "company",
    "method",
    "actual_ultimate",
    "actual_unpaid",
    "mean_ultimate_est",
    "mean_unpaid_est",
    "stddev_est",
    "cv_unpaid_est",
    "implied_pctl",
}


sns.set_theme(style="whitegrid")


def _validate_results(results: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_COLS.difference(results.columns)
    if missing:
        raise ValueError(f"Results dataframe is missing required columns: {sorted(missing)}")
    return results.copy()


def _filter_by_cv(results: pd.DataFrame, cv_limits: Tuple[float, float]) -> pd.DataFrame:
    lower, upper = cv_limits
    return results.loc[
        results["cv_unpaid_est"].between(lower, upper) & results["cv_unpaid_est"].notna()
    ]


def create_pp_plot(
    results: pd.DataFrame,
    cv_limits: Tuple[float, float] = (0.0, 1.0),
    by_line: bool = True,
) -> plt.Figure:
    """Replicate the pp-plot from the R package using matplotlib."""

    filtered = _filter_by_cv(_validate_results(results), cv_limits)
    if filtered.empty:
        raise ValueError("No rows available after applying CV filters.")

    filtered = filtered.sort_values("implied_pctl").copy()
    n = len(filtered)
    filtered["expected_pctl"] = np.arange(1, n + 1) / (n + 1)

    figure_lines = sorted(filtered["line"].unique()) if by_line else ["All"]
    n_plots = len(figure_lines)
    n_cols = min(2, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, line in enumerate(figure_lines):
        ax = axes[idx // n_cols][idx % n_cols]
        if line == "All":
            data = filtered
            title = filtered["method"].iloc[0]
        else:
            data = filtered.loc[filtered["line"] == line]
            title = f"{line}"
        if data.empty:
            ax.axis("off")
            continue
        se = 1.36 / math.sqrt(len(data))
        x = np.linspace(0, 1, 100)
        ax.plot(x, x, color="gray", linewidth=1.0)
        ax.plot(x, np.clip(x + se, 0, 1), color="gray", linestyle="--", linewidth=0.8)
        ax.plot(x, np.clip(x - se, 0, 1), color="gray", linestyle="--", linewidth=0.8)
        ax.scatter(data["expected_pctl"], data["implied_pctl"], s=20, alpha=0.8)
        ax.set_xlabel("Expected Percentile")
        ax.set_ylabel("Predicted Percentile")
        ax.set_title(f"PP Plot: {title}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for idx in range(n_plots, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    return fig


def create_histogram_plot(
    results: pd.DataFrame,
    cv_limits: Tuple[float, float] = (0.0, 1.0),
    by_line: bool = True,
    bins: Optional[int] = None,
) -> plt.Figure:
    """Replicate the implied percentile histogram from the R package."""

    filtered = _filter_by_cv(_validate_results(results), cv_limits)
    if filtered.empty:
        raise ValueError("No rows available after applying CV filters.")

    figure_lines = sorted(filtered["line"].unique()) if by_line else ["All"]
    n_plots = len(figure_lines)
    n_cols = min(2, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, line in enumerate(figure_lines):
        ax = axes[idx // n_cols][idx % n_cols]
        data = filtered if line == "All" else filtered.loc[filtered["line"] == line]
        if data.empty:
            ax.axis("off")
            continue
        ax.hist(data["implied_pctl"].dropna(), bins=bins or 10, color="#0083A9", edgecolor="white")
        ax.set_xlabel("Predicted Percentile")
        ax.set_ylabel("Count")
        title = filtered["method"].iloc[0] if line == "All" else f"{line}"
        ax.set_title(f"Histogram: {title}")
        ax.set_xlim(0, 1)
    for idx in range(n_plots, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    return fig
