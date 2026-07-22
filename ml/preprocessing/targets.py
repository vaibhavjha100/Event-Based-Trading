"""Recompute event-window returns from equity 1-min bars (source of truth)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ml.preprocess_config import WINDOW_MINUTES
from ml.preprocessing.timestamps import to_utc_timestamp


def recompute_event_returns(
    events: pd.DataFrame,
    equity_1min: pd.DataFrame,
    ohlcv_post_minutes: int = 60,
    raw_returns: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Log-returns from t0-5 to t0+window using cleaned equity minute bars.

    EOD uses the end of the available post-event window (ohlcv_post_minutes).
    Raw event_returns.csv is used only for cross-check stats.
    """
    eq = equity_1min.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, format="ISO8601")

    rows = []
    for _, ev in events.iterrows():
        eid = int(ev["event_id"])
        release = to_utc_timestamp(ev["release_time"])
        t0m5 = release - pd.Timedelta(minutes=5)
        sub = eq[eq["event_id"] == eid]
        if sub.empty:
            continue
        for asset, g in sub.groupby("asset"):
            g = g.sort_values("timestamp")
            p0 = g.loc[g["timestamp"] == t0m5, "close"]
            if p0.empty:
                continue
            p0v = float(p0.iloc[0])
            for window, mins in WINDOW_MINUTES.items():
                if window == "EOD":
                    end = release + pd.Timedelta(minutes=ohlcv_post_minutes)
                else:
                    end = release + pd.Timedelta(minutes=mins)
                p1 = g.loc[g["timestamp"] == end, "close"]
                if p1.empty:
                    # fallback: last bar at or before end
                    before = g[g["timestamp"] <= end]
                    if before.empty:
                        continue
                    p1v = float(before.iloc[-1]["close"])
                else:
                    p1v = float(p1.iloc[0])
                rows.append(
                    {
                        "event_id": eid,
                        "asset": asset,
                        "window": window,
                        "return_value": float(np.log(p1v / p0v)),
                    }
                )

    cleaned = pd.DataFrame(rows)
    crosscheck: Dict = {"compared_rows": 0, "max_abs_diff": None}
    if raw_returns is not None and not cleaned.empty:
        m = cleaned.merge(
            raw_returns,
            on=["event_id", "asset", "window"],
            how="inner",
            suffixes=("_clean", "_raw"),
        )
        if len(m):
            diffs = (m["return_value_clean"] - m["return_value_raw"]).abs()
            crosscheck = {
                "compared_rows": int(len(m)),
                "max_abs_diff": float(diffs.max()),
                "mean_abs_diff": float(diffs.mean()),
            }
    return cleaned, crosscheck


def pivot_primary_targets(event_returns: pd.DataFrame) -> pd.DataFrame:
    """Wide target columns for ML base table."""
    if event_returns.empty:
        return pd.DataFrame(columns=["event_id"])
    rows = []
    for eid, g in event_returns.groupby("event_id"):
        row = {"event_id": int(eid)}
        for _, r in g.iterrows():
            row[f"ret_{r['asset']}_{r['window']}"] = r["return_value"]
        rows.append(row)
    return pd.DataFrame(rows)
