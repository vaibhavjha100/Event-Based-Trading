"""Daily equities, event-window returns, and consistent 1-min equity OHLCV."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ml.synth_config import EQUITY_ASSETS, RETURN_WINDOWS, WINDOW_MINUTES, SynthConfig

# Starting prices
START_PRICES = {
    "SPY": 370.0,
    "QQQ": 310.0,
    "XLF": 30.0,
    "XLK": 130.0,
    "XLU": 62.0,
}

# Betas to common market factor
ASSET_BETA = {
    "SPY": 1.00,
    "QQQ": 1.20,
    "XLF": 1.05,
    "XLK": 1.25,
    "XLU": 0.55,
}


def generate_daily_equity_prices(
    events: pd.DataFrame,
    latents: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Full-horizon daily OHLCV for SPY, QQQ, and sector ETFs.

    Used for lagged controls such as ``prior_5d_SPY_return``. Mild event-day
    bumps from Z_e are optional and weak so signals remain the primary driver
    of event-window returns.
    """
    dates = pd.bdate_range(cfg.start_date, cfg.end_date, freq="C")
    n = len(dates)

    event_bump: Dict[pd.Timestamp, float] = {}
    merged = events.merge(latents, on="event_id")
    for _, row in merged.iterrows():
        d = pd.Timestamp(row["release_time"]).tz_localize(None).normalize()
        # Weak adverse → negative bump
        bump = -0.0015 * float(row["Z_e"])
        event_bump[d] = event_bump.get(d, 0.0) + bump

    market = rng.normal(0.00025, 0.009, size=n)
    for i, d in enumerate(dates):
        market[i] += event_bump.get(d.normalize(), 0.0)

    rows: List[dict] = []
    for asset in EQUITY_ASSETS:
        price = START_PRICES[asset]
        beta = ASSET_BETA[asset]
        idio = rng.normal(0.0, 0.004, size=n)
        for i, d in enumerate(dates):
            ret = beta * market[i] + idio[i]
            o = price
            c = price * np.exp(ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.002)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.002)))
            vol = float(rng.lognormal(16.5, 0.35))
            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "asset": asset,
                    "open": round(float(o), 4),
                    "high": round(float(h), 4),
                    "low": round(float(l), 4),
                    "close": round(float(c), 4),
                    "volume": round(vol, 2),
                }
            )
            price = c

    return pd.DataFrame(rows)


def generate_event_returns(
    events: pd.DataFrame,
    latents: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate event-window log-returns from signals (one-way dependency)."""
    merged = (
        events.merge(latents, on="event_id")
        .merge(signals, on="event_id")
    )
    rows: List[dict] = []

    betas = np.array(cfg.return_betas_spy, dtype=float)

    for _, row in merged.iterrows():
        s = float(row["surprise_value"])
        p = float(row["PMPS_pre_K"])
        d = float(row["disagreement_pre"])
        vix = float(row["VIX_Tminus1"])
        prior = float(row["prior_5d_SPY_return"])
        z = float(row["Z_e"])
        v_e = float(row["V_e"])
        regime = row["regime"]
        etype = row["event_type"]

        regime_m = cfg.regime_shock_mult.get(regime, 1.0)
        features = _feature_vector(s, p, d, vix, prior)

        for asset in EQUITY_ASSETS:
            sens = cfg.asset_sensitivity[asset]
            # Event-type extras
            if asset == "XLF" and etype in ("FOMC", "FED_CUTS"):
                sens *= cfg.xlf_fomc_extra
            if asset == "XLK" and etype == "FED_CUTS":
                sens *= cfg.xlk_fed_cuts_extra

            scaled = betas.copy()
            # Scale shock-related coefficients
            for idx in (1, 2, 3, 6, 7, 9, 10):
                scaled[idx] *= sens * regime_m

            r_mean = float(np.dot(scaled, features))
            # Asymmetry: adverse shocks (positive Z / positive PMPS) amplify downside
            if z > 0 or p > 0:
                if r_mean < 0:
                    r_mean *= cfg.adverse_asymmetry

            for window in RETURN_WINDOWS:
                # Longer windows accumulate more of the shock + more noise
                w_scale = {"5m": 0.35, "30m": 0.7, "60m": 1.0, "EOD": 1.35}[window]
                sigma = cfg.return_noise_base * np.sqrt(v_e) * w_scale * (1.1 if regime == "HIGH_INFLATION" else 1.0)
                # Student-t-like heavier tails via mixture
                if rng.random() < 0.08:
                    eps = rng.normal(0.0, sigma * 2.5)
                else:
                    eps = rng.normal(0.0, sigma)
                r = r_mean * w_scale + eps
                rows.append(
                    {
                        "event_id": int(row["event_id"]),
                        "asset": asset,
                        "window": window,
                        "return_value": float(r),
                    }
                )

    return pd.DataFrame(rows)


def _feature_vector(s: float, p: float, d: float, vix: float, prior: float) -> np.ndarray:
    return np.array(
        [
            1.0,          # b0
            s,            # b1
            p,            # b2
            d,            # b3
            vix,          # b4
            prior,        # b5
            s * s,        # b6
            p * p,        # b7
            vix * vix,    # b8
            s * p,        # b9
            p * p * p,    # b10
        ],
        dtype=float,
    )


def generate_equity_1min_ohlcv(
    events: pd.DataFrame,
    daily: pd.DataFrame,
    event_returns: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build event-window 1-min paths that realize ``event_returns`` targets.

    Paths hit 5m/30m/60m log-returns exactly (from t0-5). EOD returns are
    overwritten to match the path at the end of the post-event window when
    that timestamp coincides with the 60m checkpoint.

    Returns
    -------
    ohlcv : DataFrame
    event_returns_updated : DataFrame
        Same as input but with EOD ``return_value`` aligned to the path.
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    ret_idx = event_returns.set_index(["event_id", "asset", "window"])["return_value"]
    eod_updates: List[dict] = []

    rows: List[dict] = []
    for _, ev in events.iterrows():
        eid = int(ev["event_id"])
        release = pd.Timestamp(ev["release_time"])
        start = release - pd.Timedelta(minutes=cfg.ohlcv_pre_minutes)
        end = release + pd.Timedelta(minutes=cfg.ohlcv_post_minutes)
        times = pd.date_range(start, end, freq="1min", inclusive="both")

        day = release.tz_localize(None).normalize() if release.tzinfo else release.normalize()
        for asset in EQUITY_ASSETS:
            asset_daily = daily[daily["asset"] == asset].set_index("date")
            if day in asset_daily.index:
                base_price = float(asset_daily.loc[day, "open"])
            else:
                prev = asset_daily.index[asset_daily.index <= day]
                base_price = (
                    float(asset_daily.loc[prev[-1], "close"]) if len(prev) else START_PRICES[asset]
                )

            targets = {
                w: float(ret_idx.loc[(eid, asset, w)]) if (eid, asset, w) in ret_idx.index else 0.0
                for w in RETURN_WINDOWS
            }
            path_rows, eod_ret = _bridge_path(
                eid, asset, times, release, base_price, targets, cfg, rng
            )
            rows.extend(path_rows)
            eod_updates.append(
                {"event_id": eid, "asset": asset, "window": "EOD", "return_value": eod_ret}
            )

    ohlcv = pd.DataFrame(rows)
    updated = event_returns.copy()
    eod_df = pd.DataFrame(eod_updates)
    updated = updated.merge(
        eod_df,
        on=["event_id", "asset", "window"],
        how="left",
        suffixes=("", "_new"),
    )
    mask = updated["window"] == "EOD"
    updated.loc[mask, "return_value"] = updated.loc[mask, "return_value_new"]
    updated = updated.drop(columns=["return_value_new"])
    return ohlcv, updated


def _bridge_path(
    event_id: int,
    asset: str,
    times: pd.DatetimeIndex,
    release: pd.Timestamp,
    base_price: float,
    targets: Dict[str, float],
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> Tuple[List[dict], float]:
    """Construct a minute path whose cumulative returns hit window targets.

    Return windows are log-returns from ``t0-5`` to ``t0+window``.
    Also returns the realized EOD log-return at the end of the post window.
    """
    t0_minus_5 = release - pd.Timedelta(minutes=5)
    # Exact checkpoints for 5m/30m/60m only. EOD shares the post-window end with
    # 60m when ohlcv_post_minutes==60, so it is realized from the path afterward.
    checkpoints: List[Tuple[pd.Timestamp, float]] = [(t0_minus_5, 0.0)]
    for w in ("5m", "30m", "60m"):
        mins = WINDOW_MINUTES[w]
        ts = release + pd.Timedelta(minutes=mins)
        checkpoints.append((ts, targets[w]))

    checkpoints = sorted({ts: r for ts, r in checkpoints}.items(), key=lambda x: x[0])

    offsets = ((times - t0_minus_5) / pd.Timedelta(minutes=1)).astype(float)
    cp_off = np.array([((ts - t0_minus_5) / pd.Timedelta(minutes=1)) for ts, _ in checkpoints])
    cp_ret = np.array([r for _, r in checkpoints])

    cum_ret = np.interp(offsets, cp_off, cp_ret)
    noise = rng.normal(0.0, 0.00015, size=len(times))
    # Remove noise component at checkpoints so targets stay exact
    noise_at_cp = np.interp(cp_off, offsets, noise)
    noise = noise - np.interp(offsets, cp_off, noise_at_cp)
    cum_ret = cum_ret + noise

    prices = base_price * np.exp(cum_ret)
    time_list = [pd.Timestamp(t) for t in times]
    time_to_idx = {t: i for i, t in enumerate(time_list)}
    for ts, r in checkpoints:
        ts = pd.Timestamp(ts)
        if ts in time_to_idx:
            prices[time_to_idx[ts]] = base_price * np.exp(r)

    out = []
    for i, (ts, px) in enumerate(zip(time_list, prices)):
        prev = prices[i - 1] if i > 0 else px
        o = float(prev)
        c = float(px)
        noise_hl = abs(rng.normal(0, 0.0002)) * c
        h = max(o, c) + noise_hl
        l = min(o, c) - noise_hl
        vol = float(rng.lognormal(12.0, 0.4))
        out.append(
            {
                "timestamp": ts.floor("s").isoformat(),
                "event_id": int(event_id),
                "asset": asset,
                "open": round(o, 6),
                "high": round(h, 6),
                "low": round(l, 6),
                "close": round(c, 6),
                "volume": round(vol, 2),
            }
        )

    # Realized EOD = log-return from t0-5 to end of generated window
    end_ts = release + pd.Timedelta(minutes=cfg.ohlcv_post_minutes)
    p0 = base_price  # forced at t0-5
    if end_ts in time_to_idx:
        p_end = float(prices[time_to_idx[end_ts]])
    else:
        p_end = float(prices[-1])
    eod_ret = float(np.log(p_end / p0))
    return out, eod_ret
