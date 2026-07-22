"""Leakage-safe signal construction for cleaned event-level features."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def compute_event_signals(
    events: pd.DataFrame,
    band_probs: pd.DataFrame,
    band_semantics: pd.DataFrame,
    kalshi_ohlcv: pd.DataFrame,
    poly_ohlcv: pd.DataFrame,
    kalshi: pd.DataFrame,
    polymarket: pd.DataFrame,
    primary: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """Compute PMPS_pre / reaction / moments / liquidity / disagreement.

    Predictive features use only minutes_to_release in {-60, -5} (and OHLCV
    with minutes_to_release <= -5). Reaction features are separate.
    """
    # Adverse flags from band semantics (recomputed vs consensus)
    adv = band_semantics[["event_id", "platform", "band_id", "is_adverse", "band_value"]]
    # Prefer primary+research bands for signal construction
    sem_use = band_semantics[
        band_semantics["inclusion_status"].isin(["primary", "research"])
    ]
    if sem_use.empty:
        sem_use = band_semantics

    # Override is_adverse from consensus-based band semantics
    adv = sem_use[
        ["event_id", "platform", "band_id", "is_adverse"]
    ].drop_duplicates(subset=["event_id", "platform", "band_id"])
    bp = band_probs.drop(columns=["is_adverse"], errors="ignore").merge(
        adv,
        on=["event_id", "platform", "band_id"],
        how="inner",
    )

    # Track timestamps used for predictive features
    predictive_offsets_used: Set[int] = set()
    reaction_offsets_used: Set[int] = set()

    def p_bad(event_id: int, platform: str, mins: int) -> float:
        g = bp[
            (bp["event_id"] == event_id)
            & (bp["platform"] == platform)
            & (bp["minutes_to_release"] == mins)
        ]
        if g.empty:
            return np.nan
        return float(g.loc[g["is_adverse"], "probability"].sum())

    def moments_pre(event_id: int) -> dict:
        sub = bp[(bp["event_id"] == event_id) & (bp["platform"] == "KALSHI")]

        def mom(mins: int):
            g = sub[sub["minutes_to_release"] == mins]
            if g.empty:
                return 0.0, 0.0, 0.0
            x = g["band_value"].astype(float).values
            p = g["probability"].astype(float).values
            p = p / (p.sum() + 1e-12)
            mean = float(np.sum(p * x))
            var = float(np.sum(p * (x - mean) ** 2))
            skew = float(np.sum(p * (x - mean) ** 3) / (var**1.5 + 1e-12))
            return mean, var, skew

        m60, v60, s60 = mom(-60)
        m5, v5, s5 = mom(-5)
        predictive_offsets_used.update({-60, -5})
        return {
            "Delta_mean_pre": m5 - m60,
            "Delta_variance_pre": v5 - v60,
            "Delta_skew_pre": s5 - s60,
        }

    liq_k = _pre_liq_kalshi(kalshi_ohlcv, kalshi, events)
    liq_p = _pre_liq_poly(poly_ohlcv, polymarket, events)

    rows: List[dict] = []
    for _, ev in events.iterrows():
        eid = int(ev["event_id"])
        p_k_60 = p_bad(eid, "KALSHI", -60)
        p_k_5 = p_bad(eid, "KALSHI", -5)
        p_p_60 = p_bad(eid, "POLYMARKET", -60)
        p_p_5 = p_bad(eid, "POLYMARKET", -5)
        predictive_offsets_used.update({-60, -5})

        p_k_m30 = p_bad(eid, "KALSHI", -30)
        p_k_p30 = p_bad(eid, "KALSHI", 30)
        p_p_m30 = p_bad(eid, "POLYMARKET", -30)
        p_p_p30 = p_bad(eid, "POLYMARKET", 30)
        reaction_offsets_used.update({-30, 30})

        pmps_k = p_k_5 - p_k_60 if np.isfinite(p_k_5) and np.isfinite(p_k_60) else np.nan
        pmps_p = p_p_5 - p_p_60 if np.isfinite(p_p_5) and np.isfinite(p_p_60) else np.nan
        rx_k = (
            p_k_p30 - p_k_m30
            if np.isfinite(p_k_p30) and np.isfinite(p_k_m30)
            else np.nan
        )
        rx_p = (
            p_p_p30 - p_p_m30
            if np.isfinite(p_p_p30) and np.isfinite(p_p_m30)
            else np.nan
        )

        moms = moments_pre(eid)
        dv_k = float(liq_k.get(eid, {}).get("delta_volume", 0.0))
        do_k = float(liq_k.get(eid, {}).get("delta_oi", 0.0))
        dv_p = float(liq_p.get(eid, {}).get("delta_volume", 0.0))
        g_k = float(np.clip(np.log1p(max(dv_k, 0.0)) / 10.0, 0.25, 2.0))
        g_p = float(np.clip(np.log1p(max(dv_p, 0.0)) / 10.0, 0.25, 2.0))

        disagreement = (
            abs(p_k_5 - p_p_5) if np.isfinite(p_k_5) and np.isfinite(p_p_5) else np.nan
        )

        rows.append(
            {
                "event_id": eid,
                "p_bad_Kalshi_t0_60": p_k_60,
                "p_bad_Kalshi_t0_5": p_k_5,
                "p_bad_Polymarket_t0_60": p_p_60,
                "p_bad_Polymarket_t0_5": p_p_5,
                "PMPS_pre_Kalshi": pmps_k,
                "PMPS_pre_Polymarket": pmps_p,
                "PMPS_pre_weighted_Kalshi": pmps_k * g_k if np.isfinite(pmps_k) else np.nan,
                "PMPS_pre_weighted_Polymarket": pmps_p * g_p if np.isfinite(pmps_p) else np.nan,
                "Delta_mean_pre": moms["Delta_mean_pre"],
                "Delta_variance_pre": moms["Delta_variance_pre"],
                "Delta_skew_pre": moms["Delta_skew_pre"],
                "disagreement_pre": disagreement,
                "delta_volume_pre_Kalshi": dv_k,
                "delta_volume_pre_Polymarket": dv_p,
                "delta_oi_pre_Kalshi": do_k,
                # Reaction (non-predictive)
                "reaction_PMPS_Kalshi": rx_k,
                "reaction_PMPS_Polymarket": rx_p,
                "reaction_p_bad_Kalshi_t0_m30": p_k_m30,
                "reaction_p_bad_Kalshi_t0_p30": p_k_p30,
                "reaction_p_bad_Polymarket_t0_m30": p_p_m30,
                "reaction_p_bad_Polymarket_t0_p30": p_p_p30,
            }
        )

    meta = {
        "predictive_minutes_to_release_used": sorted(predictive_offsets_used),
        "reaction_minutes_to_release_used": sorted(reaction_offsets_used),
        "predictive_max_offset": max(predictive_offsets_used) if predictive_offsets_used else None,
    }
    return pd.DataFrame(rows), meta


def _pre_liq_kalshi(ohlcv, contracts, events) -> dict:
    from ml.preprocessing.timestamps import to_utc_timestamp

    core = contracts[contracts["edge_case_type"] == "none"]
    if ohlcv.empty or core.empty:
        return {}
    ticker_to_event = core.set_index("ticker")["event_id"].dropna().astype(int).to_dict()
    release_map = {
        int(r.event_id): to_utc_timestamp(r.release_time) for r in events.itertuples()
    }
    df = ohlcv.copy()
    df["event_id"] = df["ticker"].map(ticker_to_event)
    df = df.dropna(subset=["event_id"])
    df["event_id"] = df["event_id"].astype(int)
    out = {}
    for eid, g in df.groupby("event_id"):
        release = release_map.get(eid)
        if release is None:
            continue
        pre = g[
            (g["timestamp"] >= release - pd.Timedelta(minutes=60))
            & (g["timestamp"] <= release - pd.Timedelta(minutes=5))
        ].sort_values("timestamp")
        if pre.empty:
            out[eid] = {"delta_volume": 0.0, "delta_oi": 0.0}
            continue
        early, late = pre.head(10), pre.tail(10)
        out[eid] = {
            "delta_volume": float(late["volume"].sum() - early["volume"].sum()),
            "delta_oi": float(
                late["open_interest"].mean() - early["open_interest"].mean()
            )
            if "open_interest" in late.columns
            else 0.0,
        }
    return out


def _pre_liq_poly(ohlcv, contracts, events) -> dict:
    from ml.preprocessing.timestamps import to_utc_timestamp

    core = contracts[
        (contracts["edge_case_type"] == "none") & (contracts["band_id"].notna())
    ]
    if ohlcv.empty or core.empty:
        return {}
    cid_to_event = core.set_index("condition_id")["event_id"].dropna().astype(int).to_dict()
    release_map = {
        int(r.event_id): to_utc_timestamp(r.release_time) for r in events.itertuples()
    }
    df = ohlcv.copy()
    df["event_id"] = df["condition_id"].map(cid_to_event)
    df = df.dropna(subset=["event_id"])
    df["event_id"] = df["event_id"].astype(int)
    out = {}
    for eid, g in df.groupby("event_id"):
        release = release_map.get(eid)
        if release is None:
            continue
        pre = g[
            (g["timestamp"] >= release - pd.Timedelta(minutes=60))
            & (g["timestamp"] <= release - pd.Timedelta(minutes=5))
        ].sort_values("timestamp")
        if pre.empty:
            out[eid] = {"delta_volume": 0.0}
            continue
        early, late = pre.head(10), pre.tail(10)
        out[eid] = {"delta_volume": float(late["volume"].sum() - early["volume"].sum())}
    return out
