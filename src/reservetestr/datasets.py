"""Data loading and triangle preparation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chainladder as cl
import pandas as pd

LOSS_TYPE_COLUMN_MAP = {
    "paid": "cum_paid_loss",
    "case": "cum_caseinc_loss",
    "ultimate": "cum_incurred_loss",
}

MEYERS_TRIANGLES_FILE = resources.files("reservetestr.data").joinpath(
    "meyers_triangles_long.csv"
)


@dataclass(frozen=True)
class TriangleRecord:
    """Holds train/test triangles for a single company and line."""

    line: str
    group_id: int
    company: str
    train_triangles: Dict[str, Optional[cl.Triangle]]
    test_triangles: Dict[str, Optional[cl.Triangle]]
    actual_ultimates: Dict[str, float] = field(default_factory=dict)


def _chainladder_clrd_path() -> Path:
    import chainladder.utils as utils

    return Path(utils.__file__).resolve().parent / "data" / "clrd.csv"


def load_clrd_dataframe(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the CLRD dataset bundled with chainladder."""

    csv_path = Path(path) if path else _chainladder_clrd_path()
    df = pd.read_csv(csv_path)
    renamed = df.rename(
        columns={
            "GRCODE": "group_id",
            "GRNAME": "company",
            "AccidentYear": "accident_year",
            "DevelopmentYear": "development_year",
            "DevelopmentLag": "development_lag",
            "IncurLoss": "cum_incurred_loss",
            "CumPaidLoss": "cum_paid_loss",
            "BulkLoss": "bulk_ibnr",
            "EarnedPremDIR": "direct_ep",
            "EarnedPremCeded": "ceded_ep",
            "EarnedPremNet": "net_ep",
            "Single": "single_entity",
            "PostedReserve97": "posted_reserve_1997",
            "LOB": "line",
        }
    )
    renamed["line"] = renamed["line"].str.lower()
    renamed["cum_caseinc_loss"] = renamed["cum_incurred_loss"] - renamed["bulk_ibnr"]
    renamed["booked_ult_loss"] = renamed["cum_incurred_loss"]
    renamed["calendar_year"] = (
        renamed["accident_year"] + renamed["development_lag"] - 1
    )
    renamed = renamed.sort_values(
        ["line", "group_id", "accident_year", "development_lag"]
    )
    return renamed.reset_index(drop=True)


def _subset_path() -> Path:
    return resources.files("reservetestr.data").joinpath(
        "clrd_triangles_fortesting_list.csv"
    )


def _load_subset_pairs(subset_path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = Path(subset_path) if subset_path else _subset_path()
    subset = pd.read_csv(csv_path)
    subset["line"] = subset["line"].str.lower()
    subset["group_id"] = subset["group_id"].astype(int)
    return subset


def load_meyers_subset(
    df: Optional[pd.DataFrame] = None,
    subset_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Filter CLRD data down to the Meyers testing subset."""

    base_df = df.copy() if df is not None else load_clrd_dataframe()
    subset = _load_subset_pairs(subset_path)
    merged = base_df.merge(subset, on=["line", "group_id"], how="inner")
    return merged


def build_triangle_records(
    df: Optional[pd.DataFrame] = None,
    subset_only: bool = True,
    evaluation_year: Optional[int] = None,
    use_r_actuals: bool = True,
) -> List[TriangleRecord]:
    """Create `TriangleRecord` objects from the CLRD data."""

    base_df = df.copy() if df is not None else load_clrd_dataframe()
    if subset_only:
        base_df = load_meyers_subset(base_df)

    if evaluation_year is None:
        evaluation_year = int(base_df["accident_year"].max())

    actual_lookup = _load_actual_ultimate_lookup() if use_r_actuals else {}

    records: List[TriangleRecord] = []
    grouped = base_df.groupby(["line", "group_id", "company"], sort=True)
    for (line, group_id, company), group_df in grouped:
        train_df, test_df = _split_long_triangle(group_df, evaluation_year)
        train_triangles: Dict[str, Optional[cl.Triangle]] = {}
        test_triangles: Dict[str, Optional[cl.Triangle]] = {}
        for loss_type, column in LOSS_TYPE_COLUMN_MAP.items():
            train_triangles[loss_type] = _make_triangle(train_df, column)
            test_triangles[loss_type] = _make_triangle(test_df, column)
        actuals = actual_lookup.get((line, int(group_id), company), {})
        records.append(
            TriangleRecord(
                line=line,
                group_id=int(group_id),
                company=company,
                train_triangles=train_triangles,
                test_triangles=test_triangles,
                actual_ultimates=actuals,
            )
        )
    return records


def _split_long_triangle(
    group_df: pd.DataFrame,
    evaluation_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = group_df.copy()
    df["calendar_year"] = df["calendar_year"].fillna(
        df["accident_year"] + df["development_lag"] - 1
    )
    max_dev = int(df["development_lag"].max())
    train_mask = df["calendar_year"] <= evaluation_year
    test_mask = (df["calendar_year"] > evaluation_year) | (
        df["development_lag"] == max_dev
    )
    return df.loc[train_mask], df.loc[test_mask]


def _make_triangle(df: pd.DataFrame, value_column: str) -> Optional[cl.Triangle]:
    if df.empty:
        return None
    tri_df = df[["accident_year", "development_lag", value_column]].rename(
        columns={
            "accident_year": "origin",
            "development_lag": "development_lag",
            value_column: "values",
        }
    )
    origin_years = tri_df["origin"].astype(int)
    tri_df["origin"] = pd.PeriodIndex(origin_years, freq="Y")
    dev_years = origin_years + tri_df["development_lag"] - 1
    tri_df["development"] = pd.PeriodIndex(dev_years.astype(int), freq="Y")
    triangle = cl.Triangle(
        data=tri_df[["origin", "development", "values"]],
        origin="origin",
        development="development",
        columns=["values"],
        cumulative=True,
    )
    return triangle["values"]


def _load_actual_ultimate_lookup() -> Dict[Tuple[str, int, str], Dict[str, float]]:
    df = pd.read_csv(MEYERS_TRIANGLES_FILE)
    df["line"] = df["line"].str.lower()
    df["loss_type"] = df["loss_type"].str.lower()
    df["segment"] = df["segment"].str.lower()
    actuals = (
        df.loc[(df["segment"] == "test") & (df["development_lag"] == 10)]
        .groupby(["line", "group_id", "company", "loss_type"])["value"]
        .sum()
        .reset_index()
    )
    lookup: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for row in actuals.itertuples():
        key = (row.line, int(row.group_id), row.company)
        lookup.setdefault(key, {})[row.loss_type] = float(row.value)
    return lookup
