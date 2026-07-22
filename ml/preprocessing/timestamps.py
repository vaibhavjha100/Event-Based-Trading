"""Timestamp utilities for preprocessing."""

from __future__ import annotations

import pandas as pd


def to_utc(series: pd.Series) -> pd.Series:
    """Parse timestamps to timezone-aware UTC."""
    return pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")


def to_utc_timestamp(value) -> pd.Timestamp:
    """Parse a single timestamp to UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def sort_dedupe(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sort by keys and drop exact duplicate rows."""
    out = df.sort_values(keys).drop_duplicates().reset_index(drop=True)
    return out
