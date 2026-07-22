"""Kalshi-style contract metadata and event-window 1-min OHLCV."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.generators.bands import get_event_band_defs
from ml.synth_config import SynthConfig

NOISE_TITLES = [
    ("GASOLINE", "Will average US gas price exceed $4.00?"),
    ("TSA", "Will TSA throughput exceed 2.5M on Friday?"),
    ("AIPOLICY", "Will Congress pass an AI safety bill this quarter?"),
    ("CRYPTOETF", "Will a new crypto ETF launch this month?"),
]


def _month_code(ts: pd.Timestamp) -> str:
    return ts.strftime("%y%b").upper()


def generate_kalshi_contracts(
    events: pd.DataFrame,
    latents: pd.DataFrame,
    consensus: pd.DataFrame,
    band_probs: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate Kalshi contract metadata including edge cases."""
    band_defs = get_event_band_defs(events, consensus)
    merged = events.merge(latents, on="event_id")
    rows: List[dict] = []

    # Pre-index last pre-release probs for last_price
    pre = band_probs[band_probs["minutes_to_release"] == -5]
    pre_k = pre[pre["platform"] == "KALSHI"]

    for _, row in merged.iterrows():
        if row["event_type"] == "NOISE":
            continue
        eid = int(row["event_id"])
        release = pd.Timestamp(row["release_time"])
        event_ticker = f"{row['event_type']}-{_month_code(release)}-E{eid}"
        open_time = release - pd.Timedelta(days=int(rng.integers(5, 14)))
        close_time = release + pd.Timedelta(hours=2)
        created = open_time - pd.Timedelta(days=1)
        l_e = float(row["L_e"])
        z_e = float(row["Z_e"])

        for band_id, value, label, _is_adv in band_defs[eid]:
            ticker = f"{event_ticker}-B{band_id}"
            prob_row = pre_k[(pre_k["event_id"] == eid) & (pre_k["band_id"] == band_id)]
            last_price = float(prob_row["probability"].iloc[0]) if len(prob_row) else 0.2
            spread = max(0.01, 0.04 / np.sqrt(l_e))
            yes_bid = max(0.01, last_price - spread / 2)
            yes_ask = min(0.99, last_price + spread / 2)
            volume = float(rng.lognormal(8.0 + 0.3 * np.log(l_e), 0.4) * (1 + abs(z_e)))
            oi = float(volume * rng.uniform(0.4, 0.9))

            # Resolve: highest post-release probability band wins
            post = band_probs[
                (band_probs["event_id"] == eid)
                & (band_probs["platform"] == "KALSHI")
                & (band_probs["minutes_to_release"] == 30)
            ]
            if len(post):
                winner = int(post.loc[post["probability"].idxmax(), "band_id"])
                result = "yes" if band_id == winner else "no"
            else:
                result = "no"

            rows.append(
                {
                    "ticker": ticker,
                    "event_ticker": event_ticker,
                    "market_type": "binary",
                    "yes_sub_title": label,
                    "title": f"{row['macro_variable']} outcome: {label}?",
                    "status": "resolved",
                    "last_price": round(last_price, 4),
                    "yes_bid": round(yes_bid, 4),
                    "yes_ask": round(yes_ask, 4),
                    "no_bid": round(1 - yes_ask, 4),
                    "no_ask": round(1 - yes_bid, 4),
                    "volume": round(volume, 2),
                    "volume_24h": round(volume * rng.uniform(0.15, 0.4), 2),
                    "open_interest": round(oi, 2),
                    "result": result,
                    "created_time": created.isoformat(),
                    "open_time": open_time.isoformat(),
                    "close_time": close_time.isoformat(),
                    "event_id": eid,
                    "band_id": band_id,
                    "edge_case_type": "none",
                }
            )

    # Edge cases
    rows.extend(_edge_case_kalshi(events, cfg, rng))
    return pd.DataFrame(rows)


def _edge_case_kalshi(
    events: pd.DataFrame, cfg: SynthConfig, rng: np.random.Generator
) -> List[dict]:
    rows: List[dict] = []
    start = pd.Timestamp(events["release_time"].min())
    end = pd.Timestamp(events["release_time"].max())

    # Noise contracts
    for i in range(cfg.n_noise_contracts):
        title_key, title = NOISE_TITLES[i % len(NOISE_TITLES)]
        mid = start + (end - start) * rng.random()
        open_time = mid - pd.Timedelta(days=7)
        close_time = mid + pd.Timedelta(days=1)
        ticker = f"NOISE-{title_key}-{i}"
        last_price = float(rng.uniform(0.2, 0.8))
        rows.append(
            _kalshi_edge_row(
                ticker=ticker,
                event_ticker=f"NOISE-{title_key}",
                title=title,
                yes_sub_title="Yes",
                last_price=last_price,
                open_time=open_time,
                close_time=close_time,
                event_id=None,
                band_id=None,
                edge_case_type="noise",
                rng=rng,
            )
        )

    # Malformed band structures
    for i in range(cfg.n_malformed_contracts):
        # Pick a CPI-like event to attach malformed bands near
        cpi = events[events["event_type"] == "CPI"]
        if len(cpi) == 0:
            break
        base = cpi.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        release = pd.Timestamp(base["release_time"])
        eid = int(base["event_id"])
        event_ticker = f"MALFORM-CPI-{eid}-{i}"
        # Overlapping / non-monotonic thresholds
        malformed = [
            (0, 3.5, "Above 3.5%"),
            (1, 3.0, "Above 3.0%"),  # non-monotonic vs previous
            (2, 3.2, "3.0% to 3.5%"),  # overlapping
        ]
        for band_id, _val, label in malformed:
            ticker = f"{event_ticker}-B{band_id}"
            rows.append(
                _kalshi_edge_row(
                    ticker=ticker,
                    event_ticker=event_ticker,
                    title=f"Malformed CPI bands (edge case): {label}",
                    yes_sub_title=label,
                    last_price=float(rng.uniform(0.1, 0.5)),
                    open_time=release - pd.Timedelta(days=5),
                    close_time=release + pd.Timedelta(hours=2),
                    event_id=eid,
                    band_id=band_id,
                    edge_case_type="malformed_bands",
                    rng=rng,
                )
            )

    # Timestamp-overlap but unlinked
    macro = events[events["event_type"].isin(["CPI", "FOMC"])]
    for i in range(cfg.n_overlap_unlinked_contracts):
        if len(macro) == 0:
            break
        base = macro.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        release = pd.Timestamp(base["release_time"])
        eid = int(base["event_id"])
        ticker = f"OVERLAP-UNLINKED-{eid}-{i}"
        rows.append(
            _kalshi_edge_row(
                ticker=ticker,
                event_ticker=f"OVERLAP-{eid}",
                title="Will celebrity X announce something today?",
                yes_sub_title="Yes",
                last_price=float(rng.uniform(0.15, 0.6)),
                open_time=release - pd.Timedelta(hours=6),
                close_time=release + pd.Timedelta(hours=6),
                event_id=eid,
                band_id=0,
                edge_case_type="timestamp_overlap_unlinked",
                rng=rng,
            )
        )

    return rows


def _kalshi_edge_row(
    *,
    ticker: str,
    event_ticker: str,
    title: str,
    yes_sub_title: str,
    last_price: float,
    open_time: pd.Timestamp,
    close_time: pd.Timestamp,
    event_id: Optional[int],
    band_id: Optional[int],
    edge_case_type: str,
    rng: np.random.Generator,
) -> dict:
    spread = 0.03
    yes_bid = max(0.01, last_price - spread / 2)
    yes_ask = min(0.99, last_price + spread / 2)
    volume = float(rng.lognormal(6.5, 0.5))
    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "market_type": "binary",
        "yes_sub_title": yes_sub_title,
        "title": title,
        "status": "closed",
        "last_price": round(last_price, 4),
        "yes_bid": round(yes_bid, 4),
        "yes_ask": round(yes_ask, 4),
        "no_bid": round(1 - yes_ask, 4),
        "no_ask": round(1 - yes_bid, 4),
        "volume": round(volume, 2),
        "volume_24h": round(volume * 0.25, 2),
        "open_interest": round(volume * 0.5, 2),
        "result": "yes" if last_price > 0.5 else "no",
        "created_time": (open_time - pd.Timedelta(days=1)).isoformat(),
        "open_time": open_time.isoformat(),
        "close_time": close_time.isoformat(),
        "event_id": event_id if event_id is not None else pd.NA,
        "band_id": band_id if band_id is not None else pd.NA,
        "edge_case_type": edge_case_type,
    }


def generate_kalshi_ohlcv(
    contracts: pd.DataFrame,
    events: pd.DataFrame,
    latents: pd.DataFrame,
    band_probs: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate event-window 1-min YES price paths for Kalshi contracts."""
    event_map = events.set_index("event_id")
    latent_map = latents.set_index("event_id")
    rows: List[dict] = []

    # Index probs for interpolation anchors
    for _, c in contracts.iterrows():
        ticker = c["ticker"]
        edge = c["edge_case_type"]
        if edge == "noise" or pd.isna(c["event_id"]):
            # Independent random walk around last_price
            mid = pd.Timestamp(c["open_time"]) + (
                pd.Timestamp(c["close_time"]) - pd.Timestamp(c["open_time"])
            ) / 2
            path = _noise_path(mid, float(c["last_price"]), cfg, rng)
            rows.extend(_bars_from_path(ticker, path, float(c["open_interest"]), rng))
            continue

        eid = int(c["event_id"])
        if eid not in event_map.index:
            continue
        release = pd.Timestamp(event_map.loc[eid, "release_time"])
        band_id = int(c["band_id"]) if not pd.isna(c["band_id"]) else 0
        l_e = float(latent_map.loc[eid, "L_e"]) if eid in latent_map.index else 1.0
        v_e = float(latent_map.loc[eid, "V_e"]) if eid in latent_map.index else 1.0

        if edge in ("malformed_bands", "timestamp_overlap_unlinked"):
            # Unlinked to Z_e: mild noise around last_price near release window
            path = _noise_path(release, float(c["last_price"]), cfg, rng, scale=0.02)
            rows.extend(_bars_from_path(ticker, path, float(c["open_interest"]), rng))
            continue

        anchors = _prob_anchors(band_probs, eid, "KALSHI", band_id, release)
        path = _anchored_path(release, anchors, v_e, l_e, cfg, rng)
        base_oi = float(c["open_interest"])
        rows.extend(_bars_from_path(ticker, path, base_oi, rng, l_e=l_e, v_e=v_e))

    return pd.DataFrame(rows)


def _prob_anchors(
    band_probs: pd.DataFrame,
    event_id: int,
    platform: str,
    band_id: int,
    release: pd.Timestamp,
) -> Dict[int, float]:
    sub = band_probs[
        (band_probs["event_id"] == event_id)
        & (band_probs["platform"] == platform)
        & (band_probs["band_id"] == band_id)
    ]
    anchors: Dict[int, float] = {}
    for _, r in sub.iterrows():
        anchors[int(r["minutes_to_release"])] = float(r["probability"])
    # Ensure endpoints for window
    if -60 not in anchors and -5 in anchors:
        anchors[-120] = max(0.01, anchors[-5] - 0.05)
    return anchors


def _anchored_path(
    release: pd.Timestamp,
    anchors: Dict[int, float],
    v_e: float,
    l_e: float,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    start = release - pd.Timedelta(minutes=cfg.ohlcv_pre_minutes)
    end = release + pd.Timedelta(minutes=cfg.ohlcv_post_minutes)
    times = pd.date_range(start, end, freq="1min", inclusive="both")
    offsets = ((times - release) / pd.Timedelta(minutes=1)).astype(int)

    sorted_keys = sorted(anchors.keys())
    prices = []
    for off in offsets:
        # Piecewise linear interp in offset space
        if off <= sorted_keys[0]:
            p = anchors[sorted_keys[0]]
        elif off >= sorted_keys[-1]:
            p = anchors[sorted_keys[-1]]
        else:
            lo = max(k for k in sorted_keys if k <= off)
            hi = min(k for k in sorted_keys if k >= off)
            if lo == hi:
                p = anchors[lo]
            else:
                w = (off - lo) / (hi - lo)
                p = (1 - w) * anchors[lo] + w * anchors[hi]
        noise = rng.normal(0.0, 0.01 * np.sqrt(v_e) / np.sqrt(max(l_e, 0.2)))
        # Jump near release
        if abs(off) <= 2:
            noise += rng.normal(0.0, 0.03 * np.sqrt(v_e))
        prices.append(float(np.clip(p + noise, 0.01, 0.99)))

    return pd.DataFrame({"timestamp": times, "close": prices})


def _noise_path(
    center: pd.Timestamp,
    last_price: float,
    cfg: SynthConfig,
    rng: np.random.Generator,
    scale: float = 0.015,
) -> pd.DataFrame:
    start = center - pd.Timedelta(minutes=cfg.ohlcv_pre_minutes)
    end = center + pd.Timedelta(minutes=cfg.ohlcv_post_minutes)
    times = pd.date_range(start, end, freq="1min", inclusive="both")
    n = len(times)
    shocks = rng.normal(0.0, scale, size=n)
    path = last_price + np.cumsum(shocks)
    path = np.clip(path - path[n // 2] + last_price, 0.01, 0.99)
    return pd.DataFrame({"timestamp": times, "close": path})


def _bars_from_path(
    ticker: str,
    path: pd.DataFrame,
    base_oi: float,
    rng: np.random.Generator,
    l_e: float = 1.0,
    v_e: float = 1.0,
) -> List[dict]:
    closes = path["close"].values
    times = path["timestamp"].values
    rows = []
    oi = base_oi
    for i, (ts, c) in enumerate(zip(times, closes)):
        prev = closes[i - 1] if i > 0 else c
        o = prev
        noise = abs(rng.normal(0, 0.002 * np.sqrt(v_e)))
        h = min(0.99, max(o, c) + noise)
        l = max(0.01, min(o, c) - noise)
        vol = float(rng.lognormal(2.0 + 0.2 * np.log(l_e), 0.5) * (1 + 0.3 * v_e))
        oi = max(1.0, oi + rng.normal(0, vol * 0.05))
        rows.append(
            {
                "timestamp": pd.Timestamp(ts).floor("s").isoformat(),
                "ticker": ticker,
                "open": round(float(o), 4),
                "high": round(float(h), 4),
                "low": round(float(l), 4),
                "close": round(float(c), 4),
                "volume": round(vol, 2),
                "open_interest": round(float(oi), 2),
            }
        )
    return rows
