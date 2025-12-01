"""Backtesting orchestration."""
from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Optional, Sequence

import pandas as pd

from .datasets import TriangleRecord

LOGGER = logging.getLogger(__name__)

RESULT_COLUMNS = [
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
]


def run_single_backtest(
    triangle_records: Sequence[TriangleRecord],
    reserving_function: Callable,
    lines_to_include: Optional[Iterable[str]] = None,
    loss_type: str = "paid",
    method_label: Optional[str] = None,
    on_error: str = "warn",
    **kwargs,
) -> pd.DataFrame:
    """Apply a reserving function record-by-record and collect the results."""

    lines_filter = {line.lower() for line in lines_to_include} if lines_to_include else None
    method_name = method_label or getattr(reserving_function, "__name__", "method")

    rows: List[dict] = []
    for record in triangle_records:
        if lines_filter and record.line.lower() not in lines_filter:
            continue
        try:
            result = reserving_function(
                record.train_triangles,
                record.test_triangles,
                loss_type=loss_type,
                actual_ultimates=record.actual_ultimates,
                **kwargs,
            )
        except Exception as exc:  # pylint: disable=broad-except
            if on_error == "raise":
                raise
            LOGGER.warning(
                "Failed to evaluate %s for %s-%s: %s",
                method_name,
                record.line,
                record.group_id,
                exc,
            )
            continue
        if not result:
            continue
        row = {
            "line": record.line,
            "group_id": record.group_id,
            "company": record.company,
            "method": method_name,
        }
        row.update(result)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.DataFrame(rows)
    return df[RESULT_COLUMNS]
