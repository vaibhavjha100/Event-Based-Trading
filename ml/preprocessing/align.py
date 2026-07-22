"""Event alignment helpers (consensus surprise, controls)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.preprocessing.band_semantics import latest_prerelease_consensus
from ml.preprocessing.timestamps import to_utc_timestamp


def build_event_consensus_features(
    events: pd.DataFrame, consensus: pd.DataFrame
) -> pd.DataFrame:
    """Per-event latest pre-release consensus, actual, and surprise."""
    rows = []
    for _, ev in events.iterrows():
        eid = int(ev["event_id"])
        cons = latest_prerelease_consensus(consensus, eid, ev["release_time"])
        if cons is None:
            rows.append(
                {
                    "event_id": eid,
                    "consensus_value": np.nan,
                    "consensus_time": pd.NaT,
                    "actual_value": np.nan,
                    "surprise": np.nan,
                }
            )
            continue
        actual = float(cons["actual_value"])
        cval = float(cons["consensus_value"])
        rows.append(
            {
                "event_id": eid,
                "consensus_value": cval,
                "consensus_time": cons["consensus_time"],
                "actual_value": actual,
                "surprise": actual - cval,
            }
        )
    return pd.DataFrame(rows)


def build_controls(
    events: pd.DataFrame, vix: pd.DataFrame, daily_equity: pd.DataFrame
) -> pd.DataFrame:
    """VIX_Tminus1 and prior_5d_SPY_return from cleaned daily series."""
    vix = vix.copy()
    vix["date"] = pd.to_datetime(vix["date"])
    spy = daily_equity[daily_equity["asset"] == "SPY"].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy["prior_5d_SPY_return"] = np.log(spy["close"] / spy["close"].shift(5))

    rows = []
    for _, ev in events.iterrows():
        eid = int(ev["event_id"])
        release = to_utc_timestamp(ev["release_time"])
        day = release.tz_localize(None).normalize()

        vprev = vix[vix["date"] < day]
        vix_tm1 = float(vprev["vix_close"].iloc[-1]) if len(vprev) else np.nan

        sprev = spy[spy["date"] < day]
        prior = 0.0
        if len(sprev):
            val = sprev["prior_5d_SPY_return"].iloc[-1]
            prior = 0.0 if pd.isna(val) else float(val)

        rows.append(
            {
                "event_id": eid,
                "VIX_Tminus1": vix_tm1,
                "prior_5d_SPY_return": prior,
                "regime": ev.get("regime"),
                "event_type": ev.get("event_type"),
                "release_time": release,
            }
        )
    out = pd.DataFrame(rows)
    for reg in ("HIGH_INFLATION", "NORMAL", "EASING", "TIGHTENING"):
        out[f"regime_{reg}"] = (out["regime"] == reg).astype(int)
    for et in ("CPI", "FOMC", "FED_CUTS", "NFP", "NOISE"):
        out[f"event_type_{et}"] = (out["event_type"] == et).astype(int)
    return out


def split_pm_ohlcv_windows(
    kalshi_ohlcv: pd.DataFrame,
    poly_ohlcv: pd.DataFrame,
    events: pd.DataFrame,
    kalshi: pd.DataFrame,
    polymarket: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split PM OHLCV into pre-release (<= t0-5) and reaction (t0-30..t0+30) tables."""
    release_map = {
        int(r.event_id): to_utc_timestamp(r.release_time) for r in events.itertuples()
    }
    k_eid = kalshi.set_index("ticker")["event_id"].to_dict()
    p_eid = polymarket.set_index("condition_id")["event_id"].to_dict()

    pre_rows = []
    rx_rows = []

    k = kalshi_ohlcv.copy()
    k["event_id"] = k["ticker"].map(k_eid)
    k = k.dropna(subset=["event_id"])
    k["event_id"] = k["event_id"].astype(int)
    for eid, g in k.groupby("event_id"):
        t0 = release_map.get(eid)
        if t0 is None:
            continue
        g = g.copy()
        g["minutes_to_release"] = (
            (g["timestamp"] - t0) / pd.Timedelta(minutes=1)
        ).astype(int)
        pre = g[g["minutes_to_release"] <= -5]
        rx = g[(g["minutes_to_release"] >= -30) & (g["minutes_to_release"] <= 30)]
        pre_rows.append(pre.assign(platform="KALSHI", contract_id=pre["ticker"]))
        rx_rows.append(rx.assign(platform="KALSHI", contract_id=rx["ticker"]))

    p = poly_ohlcv.copy()
    p["event_id"] = p["condition_id"].map(p_eid)
    p = p.dropna(subset=["event_id"])
    p["event_id"] = p["event_id"].astype(int)
    for eid, g in p.groupby("event_id"):
        t0 = release_map.get(eid)
        if t0 is None:
            continue
        g = g.copy()
        g["minutes_to_release"] = (
            (g["timestamp"] - t0) / pd.Timedelta(minutes=1)
        ).astype(int)
        pre = g[g["minutes_to_release"] <= -5]
        rx = g[(g["minutes_to_release"] >= -30) & (g["minutes_to_release"] <= 30)]
        pre_rows.append(pre.assign(platform="POLYMARKET", contract_id=pre["condition_id"]))
        rx_rows.append(rx.assign(platform="POLYMARKET", contract_id=rx["condition_id"]))

    pre_df = pd.concat(pre_rows, ignore_index=True) if pre_rows else pd.DataFrame()
    rx_df = pd.concat(rx_rows, ignore_index=True) if rx_rows else pd.DataFrame()
    return pre_df, rx_df
