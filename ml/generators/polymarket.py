"""Polymarket-style contract metadata and event-window 1-min OHLCV."""

from __future__ import annotations

import json
import hashlib
from typing import List, Optional

import numpy as np
import pandas as pd

from ml.generators.bands import get_event_band_defs
from ml.generators.kalshi import _anchored_path, _noise_path, _prob_anchors
from ml.synth_config import SynthConfig

NOISE_QUESTIONS = [
    "Will US average gas prices exceed $4.00 this month?",
    "Will TSA checkpoint throughput exceed 2.5 million on the Friday before the holiday?",
    "Will Congress pass comprehensive AI safety legislation this quarter?",
    "Will a major exchange list a new meme-coin futures product?",
]


def _condition_id(seed_str: str) -> str:
    return "0x" + hashlib.sha256(seed_str.encode()).hexdigest()[:40]


def _clob_ids(condition_id: str, n: int = 2) -> str:
    ids = [
        str(int(hashlib.md5(f"{condition_id}-{i}".encode()).hexdigest()[:15], 16))
        for i in range(n)
    ]
    return json.dumps(ids)


def generate_polymarket_contracts(
    events: pd.DataFrame,
    latents: pd.DataFrame,
    consensus: pd.DataFrame,
    band_probs: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate Polymarket contract metadata including edge cases."""
    band_defs = get_event_band_defs(events, consensus)
    merged = events.merge(latents, on="event_id")
    rows: List[dict] = []
    next_id = 1

    pre = band_probs[
        (band_probs["minutes_to_release"] == -5) & (band_probs["platform"] == "POLYMARKET")
    ]

    for _, row in merged.iterrows():
        if row["event_type"] == "NOISE":
            continue
        eid = int(row["event_id"])
        release = pd.Timestamp(row["release_time"])
        bands = band_defs[eid]
        labels = [b[2] for b in bands]
        probs = []
        for band_id, _, _, _ in bands:
            pr = pre[(pre["event_id"] == eid) & (pre["band_id"] == band_id)]
            probs.append(float(pr["probability"].iloc[0]) if len(pr) else 1.0 / len(bands))
        # Normalize
        s = sum(probs) or 1.0
        probs = [p / s for p in probs]

        condition_id = _condition_id(f"poly-core-{eid}")
        slug = f"{row['event_type'].lower()}-{row['macro_variable'].lower()}-e{eid}"
        question = f"What will {row['macro_variable']} print for event {eid}?"
        l_e = float(row["L_e"])
        volume = float(rng.lognormal(10.0 + 0.25 * np.log(l_e), 0.45))
        liquidity = float(volume * rng.uniform(0.05, 0.2))

        # One multi-outcome market per event (Polymarket style)
        rows.append(
            {
                "id": next_id,
                "condition_id": condition_id,
                "question": question,
                "slug": slug,
                "outcomes": json.dumps(labels),
                "outcome_prices": json.dumps([round(p, 4) for p in probs]),
                "clob_token_ids": _clob_ids(condition_id, len(labels)),
                "volume": round(volume, 2),
                "liquidity": round(liquidity, 2),
                "active": False,
                "closed": True,
                "end_date": release.isoformat(),
                "created_at": (release - pd.Timedelta(days=10)).isoformat(),
                "market_maker_address": "0x" + hashlib.md5(f"mm-{eid}".encode()).hexdigest()[:40],
                "event_id": eid,
                "band_id": pd.NA,  # multi-outcome market covers all bands
                "edge_case_type": "none",
            }
        )
        next_id += 1

        # Also emit binary YES contracts per band for OHLCV alignment with band_id
        for band_id, _value, label, _adv in bands:
            cid = _condition_id(f"poly-band-{eid}-{band_id}")
            pr = pre[(pre["event_id"] == eid) & (pre["band_id"] == band_id)]
            p = float(pr["probability"].iloc[0]) if len(pr) else 0.2
            rows.append(
                {
                    "id": next_id,
                    "condition_id": cid,
                    "question": f"{question} — {label}",
                    "slug": f"{slug}-b{band_id}",
                    "outcomes": json.dumps(["Yes", "No"]),
                    "outcome_prices": json.dumps([round(p, 4), round(1 - p, 4)]),
                    "clob_token_ids": _clob_ids(cid, 2),
                    "volume": round(volume / max(len(bands), 1), 2),
                    "liquidity": round(liquidity / max(len(bands), 1), 2),
                    "active": False,
                    "closed": True,
                    "end_date": release.isoformat(),
                    "created_at": (release - pd.Timedelta(days=10)).isoformat(),
                    "market_maker_address": "0x"
                    + hashlib.md5(f"mm-{eid}-{band_id}".encode()).hexdigest()[:40],
                    "event_id": eid,
                    "band_id": band_id,
                    "edge_case_type": "none",
                }
            )
            next_id += 1

    edge_rows, next_id = _edge_case_polymarket(events, cfg, rng, next_id)
    rows.extend(edge_rows)
    return pd.DataFrame(rows)


def _edge_case_polymarket(
    events: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
    next_id: int,
) -> tuple:
    rows: List[dict] = []
    start = pd.Timestamp(events["release_time"].min())
    end = pd.Timestamp(events["release_time"].max())

    for i in range(cfg.n_noise_contracts):
        mid = start + (end - start) * rng.random()
        q = NOISE_QUESTIONS[i % len(NOISE_QUESTIONS)]
        cid = _condition_id(f"poly-noise-{i}")
        p = float(rng.uniform(0.2, 0.8))
        rows.append(
            _poly_edge_row(
                next_id,
                cid,
                q,
                f"noise-{i}",
                p,
                mid,
                None,
                None,
                "noise",
                rng,
            )
        )
        next_id += 1

    cpi = events[events["event_type"] == "CPI"]
    for i in range(cfg.n_malformed_contracts):
        if len(cpi) == 0:
            break
        base = cpi.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        eid = int(base["event_id"])
        release = pd.Timestamp(base["release_time"])
        cid = _condition_id(f"poly-malform-{eid}-{i}")
        # Malformed: overlapping outcome labels / prices that don't sum to 1
        outcomes = ["Above 3.5%", "Above 3.0%", "3.0% to 3.5%"]
        prices = [0.4, 0.45, 0.35]  # intentionally sum > 1
        rows.append(
            {
                "id": next_id,
                "condition_id": cid,
                "question": f"Malformed CPI market (edge case) near event {eid}",
                "slug": f"malform-cpi-{eid}-{i}",
                "outcomes": json.dumps(outcomes),
                "outcome_prices": json.dumps(prices),
                "clob_token_ids": _clob_ids(cid, 3),
                "volume": round(float(rng.lognormal(8, 0.5)), 2),
                "liquidity": round(float(rng.lognormal(6, 0.5)), 2),
                "active": False,
                "closed": True,
                "end_date": release.isoformat(),
                "created_at": (release - pd.Timedelta(days=5)).isoformat(),
                "market_maker_address": "0x" + hashlib.md5(f"mm-m-{i}".encode()).hexdigest()[:40],
                "event_id": eid,
                "band_id": 0,
                "edge_case_type": "malformed_bands",
            }
        )
        next_id += 1

    macro = events[events["event_type"].isin(["CPI", "FOMC"])]
    for i in range(cfg.n_overlap_unlinked_contracts):
        if len(macro) == 0:
            break
        base = macro.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        eid = int(base["event_id"])
        release = pd.Timestamp(base["release_time"])
        cid = _condition_id(f"poly-overlap-{eid}-{i}")
        p = float(rng.uniform(0.15, 0.55))
        rows.append(
            _poly_edge_row(
                next_id,
                cid,
                "Will a celebrity announce a major product today?",
                f"overlap-unlinked-{eid}-{i}",
                p,
                release,
                eid,
                0,
                "timestamp_overlap_unlinked",
                rng,
            )
        )
        next_id += 1

    return rows, next_id


def _poly_edge_row(
    id_: int,
    condition_id: str,
    question: str,
    slug: str,
    p: float,
    end: pd.Timestamp,
    event_id: Optional[int],
    band_id: Optional[int],
    edge_case_type: str,
    rng: np.random.Generator,
) -> dict:
    return {
        "id": id_,
        "condition_id": condition_id,
        "question": question,
        "slug": slug,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcome_prices": json.dumps([round(p, 4), round(1 - p, 4)]),
        "clob_token_ids": _clob_ids(condition_id, 2),
        "volume": round(float(rng.lognormal(7, 0.5)), 2),
        "liquidity": round(float(rng.lognormal(5, 0.5)), 2),
        "active": False,
        "closed": True,
        "end_date": end.isoformat(),
        "created_at": (end - pd.Timedelta(days=7)).isoformat(),
        "market_maker_address": "0x" + hashlib.md5(condition_id.encode()).hexdigest()[:40],
        "event_id": event_id if event_id is not None else pd.NA,
        "band_id": band_id if band_id is not None else pd.NA,
        "edge_case_type": edge_case_type,
    }


def generate_polymarket_ohlcv(
    contracts: pd.DataFrame,
    events: pd.DataFrame,
    latents: pd.DataFrame,
    band_probs: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate event-window 1-min YES price paths for Polymarket band contracts."""
    event_map = events.set_index("event_id")
    latent_map = latents.set_index("event_id")
    # Only band-level binary contracts (have band_id) for OHLCV
    band_contracts = contracts[contracts["band_id"].notna()].copy()
    rows: List[dict] = []

    for _, c in band_contracts.iterrows():
        cid = c["condition_id"]
        edge = c["edge_case_type"]
        prices = json.loads(c["outcome_prices"])
        last_price = float(prices[0]) if prices else 0.5

        if edge == "noise" or pd.isna(c["event_id"]):
            mid = pd.Timestamp(c["end_date"])
            path = _noise_path(mid, last_price, cfg, rng)
            rows.extend(_poly_bars(cid, path, rng))
            continue

        eid = int(c["event_id"])
        if eid not in event_map.index:
            continue
        release = pd.Timestamp(event_map.loc[eid, "release_time"])
        band_id = int(c["band_id"])
        l_e = float(latent_map.loc[eid, "L_e"]) if eid in latent_map.index else 1.0
        v_e = float(latent_map.loc[eid, "V_e"]) if eid in latent_map.index else 1.0

        if edge in ("malformed_bands", "timestamp_overlap_unlinked"):
            path = _noise_path(release, last_price, cfg, rng, scale=0.02)
            rows.extend(_poly_bars(cid, path, rng))
            continue

        anchors = _prob_anchors(band_probs, eid, "POLYMARKET", band_id, release)
        # Independent platform noise bias vs Kalshi
        path = _anchored_path(release, anchors, v_e, l_e, cfg, rng)
        path["close"] = np.clip(
            path["close"] + rng.normal(0, 0.01, size=len(path)), 0.01, 0.99
        )
        rows.extend(_poly_bars(cid, path, rng, l_e=l_e, v_e=v_e))

    return pd.DataFrame(rows)


def _poly_bars(
    condition_id: str,
    path: pd.DataFrame,
    rng: np.random.Generator,
    l_e: float = 1.0,
    v_e: float = 1.0,
) -> List[dict]:
    closes = path["close"].values
    times = path["timestamp"].values
    out = []
    for i, (ts, c) in enumerate(zip(times, closes)):
        prev = closes[i - 1] if i > 0 else c
        o = float(prev)
        noise = abs(rng.normal(0, 0.002 * np.sqrt(v_e)))
        h = min(0.99, max(o, float(c)) + noise)
        l = max(0.01, min(o, float(c)) - noise)
        vol = float(rng.lognormal(2.2 + 0.2 * np.log(l_e), 0.5))
        out.append(
            {
                "timestamp": pd.Timestamp(ts).floor("s").isoformat(),
                "condition_id": condition_id,
                "open": round(o, 4),
                "high": round(h, 4),
                "low": round(l, 4),
                "close": round(float(c), 4),
                "volume": round(vol, 2),
            }
        )
    return out
