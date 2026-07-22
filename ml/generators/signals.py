"""Derived event-level signals (strictly pre-release for predictive features)."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ml.synth_config import SynthConfig


def generate_event_signals(
    events: pd.DataFrame,
    latents: pd.DataFrame,
    consensus: pd.DataFrame,
    band_probs: pd.DataFrame,
    kalshi_ohlcv: pd.DataFrame,
    polymarket_ohlcv: pd.DataFrame,
    kalshi_contracts: pd.DataFrame,
    polymarket_contracts: pd.DataFrame,
    vix_daily: pd.DataFrame,
    daily_equity: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compute event_signals from PM probs, liquidity, VIX, and daily equities.

    Does **not** depend on event_returns or equity_1min_ohlcv.
    PMPS_pre uses only t0-60 and t0-5 (fully pre-release).
    """
    cons_latest = (
        consensus.sort_values("consensus_time")
        .groupby("event_id", as_index=False)
        .tail(1)[["event_id", "surprise_value"]]
    )

    vix = vix_daily.copy()
    vix["date"] = pd.to_datetime(vix["date"])

    spy = daily_equity[daily_equity["asset"] == "SPY"].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy["prior_5d_SPY_return"] = np.log(spy["close"] / spy["close"].shift(5))

    # Liquidity deltas from OHLCV in pre-release window
    liq_k = _pre_liquidity_kalshi(kalshi_ohlcv, kalshi_contracts, events, cfg)
    liq_p = _pre_liquidity_poly(polymarket_ohlcv, polymarket_contracts, events, cfg)

    rows: List[dict] = []
    for _, ev in events.iterrows():
        eid = int(ev["event_id"])
        release = pd.Timestamp(ev["release_time"])
        day = release.tz_localize(None).normalize() if release.tzinfo else release.normalize()

        p_bad = _p_bad_table(band_probs, eid)
        p_k_60 = p_bad.get(("KALSHI", -60), 0.5)
        p_k_5 = p_bad.get(("KALSHI", -5), 0.5)
        p_p_60 = p_bad.get(("POLYMARKET", -60), 0.5)
        p_p_5 = p_bad.get(("POLYMARKET", -5), 0.5)
        p_k_m30 = p_bad.get(("KALSHI", -30), p_k_60)
        p_k_p30 = p_bad.get(("KALSHI", 30), p_k_5)
        p_p_m30 = p_bad.get(("POLYMARKET", -30), p_p_60)
        p_p_p30 = p_bad.get(("POLYMARKET", 30), p_p_5)

        pmps_k = p_k_5 - p_k_60
        pmps_p = p_p_5 - p_p_60
        pmps_rx_k = p_k_p30 - p_k_m30
        pmps_rx_p = p_p_p30 - p_p_m30

        moments = _distributional_moments(band_probs, eid)
        disagreement = abs(p_k_5 - p_p_5)

        dv_k = float(liq_k.get(eid, {}).get("delta_volume", 0.0))
        do_k = float(liq_k.get(eid, {}).get("delta_oi", 0.0))
        dv_p = float(liq_p.get(eid, {}).get("delta_volume", 0.0))

        # Liquidity weights g(·): log1p normalized within batch later; local soft scale
        g_k = np.log1p(max(dv_k, 0.0)) / 10.0
        g_p = np.log1p(max(dv_p, 0.0)) / 10.0
        g_k = float(np.clip(g_k, 0.25, 2.0))
        g_p = float(np.clip(g_p, 0.25, 2.0))

        # VIX_Tminus1
        vix_prev = vix[vix["date"] < day]
        vix_tm1 = float(vix_prev["vix_close"].iloc[-1]) if len(vix_prev) else 18.0
        # Blend with latent V_e for consistency
        if eid in latents["event_id"].values:
            v_e = float(latents.loc[latents["event_id"] == eid, "V_e"].iloc[0])
            vix_tm1 = 0.7 * vix_tm1 + 0.3 * (12.0 + 8.0 * v_e) + float(rng.normal(0, 0.4))

        spy_prev = spy[spy["date"] < day]
        prior_5d = float(spy_prev["prior_5d_SPY_return"].iloc[-1]) if len(spy_prev) else 0.0
        if np.isnan(prior_5d):
            prior_5d = 0.0

        surprise = cons_latest.loc[cons_latest["event_id"] == eid, "surprise_value"]
        surprise_val = float(surprise.iloc[0]) if len(surprise) else 0.0

        regime = ev["regime"]
        etype = ev["event_type"]

        rows.append(
            {
                "event_id": eid,
                "PMPS_pre_K": pmps_k,
                "PMPS_pre_P": pmps_p,
                "PMPS_pre_weighted_K": pmps_k * g_k,
                "PMPS_pre_weighted_P": pmps_p * g_p,
                "PMPS_reaction_K": pmps_rx_k,
                "PMPS_reaction_P": pmps_rx_p,
                "p_bad_K_t0_60": p_k_60,
                "p_bad_K_t0_5": p_k_5,
                "p_bad_P_t0_60": p_p_60,
                "p_bad_P_t0_5": p_p_5,
                "Delta_mean_pre": moments["delta_mean"],
                "Delta_variance_pre": moments["delta_var"],
                "Delta_skew_pre": moments["delta_skew"],
                "disagreement_pre": disagreement,
                "delta_volume_pre_K": dv_k,
                "delta_volume_pre_P": dv_p,
                "delta_oi_pre_K": do_k,
                "VIX_Tminus1": vix_tm1,
                "prior_5d_SPY_return": prior_5d,
                "regime_HIGH_INFLATION": int(regime == "HIGH_INFLATION"),
                "regime_NORMAL": int(regime == "NORMAL"),
                "regime_EASING": int(regime == "EASING"),
                "regime_TIGHTENING": int(regime == "TIGHTENING"),
                "event_type_CPI": int(etype == "CPI"),
                "event_type_FOMC": int(etype == "FOMC"),
                "event_type_FED_CUTS": int(etype == "FED_CUTS"),
                "event_type_NFP": int(etype == "NFP"),
                "event_type_NOISE": int(etype == "NOISE"),
                "surprise_value": surprise_val,
            }
        )

    return pd.DataFrame(rows)


def _p_bad_table(band_probs: pd.DataFrame, event_id: int) -> dict:
    sub = band_probs[band_probs["event_id"] == event_id]
    out = {}
    if sub.empty:
        return out
    for (platform, mins), g in sub.groupby(["platform", "minutes_to_release"]):
        out[(platform, int(mins))] = float(g.loc[g["is_adverse"], "probability"].sum())
    return out


def _distributional_moments(band_probs: pd.DataFrame, event_id: int) -> dict:
    """Shift in mean/variance/skew of band_value distribution from t0-60 to t0-5 (Kalshi)."""
    sub = band_probs[
        (band_probs["event_id"] == event_id) & (band_probs["platform"] == "KALSHI")
    ]
    if sub.empty:
        return {"delta_mean": 0.0, "delta_var": 0.0, "delta_skew": 0.0}

    def moments(mins: int):
        g = sub[sub["minutes_to_release"] == mins]
        if g.empty:
            return 0.0, 0.0, 0.0
        x = g["band_value"].values.astype(float)
        p = g["probability"].values.astype(float)
        p = p / p.sum()
        mean = float(np.sum(p * x))
        var = float(np.sum(p * (x - mean) ** 2))
        skew = float(np.sum(p * (x - mean) ** 3) / (var ** 1.5 + 1e-12))
        return mean, var, skew

    m60, v60, s60 = moments(-60)
    m5, v5, s5 = moments(-5)
    return {
        "delta_mean": m5 - m60,
        "delta_var": v5 - v60,
        "delta_skew": s5 - s60,
    }


def _pre_liquidity_kalshi(
    ohlcv: pd.DataFrame,
    contracts: pd.DataFrame,
    events: pd.DataFrame,
    cfg: SynthConfig,
) -> dict:
    """Per-event pre-release Δvolume and ΔOI from Kalshi minute bars."""
    core = contracts[contracts["edge_case_type"] == "none"]
    if ohlcv.empty or core.empty:
        return {}
    ticker_to_event = core.set_index("ticker")["event_id"].to_dict()
    release_map = {
        int(r.event_id): pd.Timestamp(r.release_time) for r in events.itertuples()
    }

    out: dict = {}
    ohlcv = ohlcv.copy()
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True, format="ISO8601")
    ohlcv["event_id"] = ohlcv["ticker"].map(ticker_to_event)
    ohlcv = ohlcv.dropna(subset=["event_id"])
    ohlcv["event_id"] = ohlcv["event_id"].astype(int)

    for eid, g in ohlcv.groupby("event_id"):
        release = release_map.get(eid)
        if release is None:
            continue
        if release.tzinfo is None:
            release = release.tz_localize("UTC")
        pre = g[
            (g["timestamp"] >= release - pd.Timedelta(minutes=60))
            & (g["timestamp"] <= release - pd.Timedelta(minutes=5))
        ]
        if pre.empty:
            out[eid] = {"delta_volume": 0.0, "delta_oi": 0.0}
            continue
        # Compare first vs last 10 minutes of pre window
        pre = pre.sort_values("timestamp")
        early = pre.head(10)
        late = pre.tail(10)
        out[eid] = {
            "delta_volume": float(late["volume"].sum() - early["volume"].sum()),
            "delta_oi": float(late["open_interest"].mean() - early["open_interest"].mean()),
        }
    return out


def _pre_liquidity_poly(
    ohlcv: pd.DataFrame,
    contracts: pd.DataFrame,
    events: pd.DataFrame,
    cfg: SynthConfig,
) -> dict:
    core = contracts[
        (contracts["edge_case_type"] == "none") & (contracts["band_id"].notna())
    ]
    if ohlcv.empty or core.empty:
        return {}
    cid_to_event = core.set_index("condition_id")["event_id"].to_dict()
    release_map = {
        int(r.event_id): pd.Timestamp(r.release_time) for r in events.itertuples()
    }

    out: dict = {}
    ohlcv = ohlcv.copy()
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True, format="ISO8601")
    ohlcv["event_id"] = ohlcv["condition_id"].map(cid_to_event)
    ohlcv = ohlcv.dropna(subset=["event_id"])
    ohlcv["event_id"] = ohlcv["event_id"].astype(int)

    for eid, g in ohlcv.groupby("event_id"):
        release = release_map.get(eid)
        if release is None:
            continue
        if release.tzinfo is None:
            release = release.tz_localize("UTC")
        pre = g[
            (g["timestamp"] >= release - pd.Timedelta(minutes=60))
            & (g["timestamp"] <= release - pd.Timedelta(minutes=5))
        ]
        if pre.empty:
            out[eid] = {"delta_volume": 0.0}
            continue
        pre = pre.sort_values("timestamp")
        early = pre.head(10)
        late = pre.tail(10)
        out[eid] = {"delta_volume": float(late["volume"].sum() - early["volume"].sum())}
    return out
