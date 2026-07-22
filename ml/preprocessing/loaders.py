"""Load and standardize raw datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ml.preprocessing.timestamps import sort_dedupe, to_utc


@dataclass
class RawBundle:
    """Container for standardized raw tables."""

    events: pd.DataFrame
    consensus: pd.DataFrame
    kalshi: pd.DataFrame
    polymarket: pd.DataFrame
    band_probs: pd.DataFrame
    kalshi_ohlcv: pd.DataFrame
    poly_ohlcv: pd.DataFrame
    equity_1min: pd.DataFrame
    daily_equity: pd.DataFrame
    vix: pd.DataFrame
    event_returns_raw: Optional[pd.DataFrame]
    contract_event_map: Optional[pd.DataFrame]
    row_counts_raw: Dict[str, int]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required raw file missing: {path}")
    return pd.read_csv(path)


def _optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _clip_unit(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(0.0, 1.0)


def load_raw_bundle(raw_dir: Path) -> RawBundle:
    """Load raw CSVs, normalize timestamps, and apply basic range cleaning."""
    raw_dir = Path(raw_dir)
    counts: Dict[str, int] = {}

    events = _read_csv(raw_dir / "events.csv")
    counts["events"] = len(events)
    events["release_time"] = to_utc(events["release_time"])
    events = sort_dedupe(events, ["event_id"])

    consensus = _read_csv(raw_dir / "consensus_forecasts.csv")
    counts["consensus"] = len(consensus)
    consensus["consensus_time"] = to_utc(consensus["consensus_time"])
    consensus = sort_dedupe(consensus, ["event_id", "consensus_time"])

    kalshi = _read_csv(raw_dir / "kalshi_contracts.csv")
    counts["kalshi_contracts"] = len(kalshi)
    for col in ("created_time", "open_time", "close_time"):
        if col in kalshi.columns:
            kalshi[col] = to_utc(kalshi[col])
    for col in ("last_price", "yes_bid", "yes_ask", "no_bid", "no_ask"):
        if col in kalshi.columns:
            kalshi[col] = _clip_unit(kalshi[col])
    kalshi = sort_dedupe(kalshi, ["ticker"])

    poly = _read_csv(raw_dir / "polymarket_contracts.csv")
    counts["polymarket_contracts"] = len(poly)
    for col in ("end_date", "created_at"):
        if col in poly.columns:
            poly[col] = to_utc(poly[col])
    poly = sort_dedupe(poly, ["condition_id", "id"])

    band_probs = _read_csv(raw_dir / "event_band_probabilities.csv")
    counts["band_probs"] = len(band_probs)
    band_probs["timestamp"] = to_utc(band_probs["timestamp"])
    band_probs["probability"] = _clip_unit(band_probs["probability"])
    band_probs = sort_dedupe(
        band_probs, ["event_id", "platform", "band_id", "minutes_to_release"]
    )

    kalshi_ohlcv = _read_csv(raw_dir / "kalshi_1min_ohlcv.csv")
    counts["kalshi_ohlcv"] = len(kalshi_ohlcv)
    kalshi_ohlcv["timestamp"] = to_utc(kalshi_ohlcv["timestamp"])
    for col in ("open", "high", "low", "close"):
        kalshi_ohlcv[col] = _clip_unit(kalshi_ohlcv[col])
    kalshi_ohlcv["volume"] = pd.to_numeric(kalshi_ohlcv["volume"], errors="coerce").clip(lower=0)
    if "open_interest" in kalshi_ohlcv.columns:
        kalshi_ohlcv["open_interest"] = pd.to_numeric(
            kalshi_ohlcv["open_interest"], errors="coerce"
        ).clip(lower=0)
    kalshi_ohlcv = sort_dedupe(kalshi_ohlcv, ["ticker", "timestamp"])

    poly_ohlcv = _read_csv(raw_dir / "polymarket_1min_ohlcv.csv")
    counts["poly_ohlcv"] = len(poly_ohlcv)
    poly_ohlcv["timestamp"] = to_utc(poly_ohlcv["timestamp"])
    for col in ("open", "high", "low", "close"):
        poly_ohlcv[col] = _clip_unit(poly_ohlcv[col])
    poly_ohlcv["volume"] = pd.to_numeric(poly_ohlcv["volume"], errors="coerce").clip(lower=0)
    poly_ohlcv = sort_dedupe(poly_ohlcv, ["condition_id", "timestamp"])

    equity_1min = _read_csv(raw_dir / "equity_1min_ohlcv.csv")
    counts["equity_1min"] = len(equity_1min)
    equity_1min["timestamp"] = to_utc(equity_1min["timestamp"])
    equity_1min = sort_dedupe(equity_1min, ["event_id", "asset", "timestamp"])

    daily = _read_csv(raw_dir / "daily_equity_prices.csv")
    counts["daily_equity"] = len(daily)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = sort_dedupe(daily, ["asset", "date"])

    vix = _read_csv(raw_dir / "vix_daily.csv")
    counts["vix"] = len(vix)
    vix["date"] = pd.to_datetime(vix["date"])
    vix["vix_close"] = pd.to_numeric(vix["vix_close"], errors="coerce")
    vix = sort_dedupe(vix, ["date"])

    event_returns_raw = _optional_csv(raw_dir / "event_returns.csv")
    if event_returns_raw is not None:
        counts["event_returns_raw"] = len(event_returns_raw)

    cmap = _optional_csv(raw_dir / "contract_event_map.csv")
    if cmap is not None:
        counts["contract_event_map"] = len(cmap)

    return RawBundle(
        events=events,
        consensus=consensus,
        kalshi=kalshi,
        polymarket=poly,
        band_probs=band_probs,
        kalshi_ohlcv=kalshi_ohlcv,
        poly_ohlcv=poly_ohlcv,
        equity_1min=equity_1min,
        daily_equity=daily,
        vix=vix,
        event_returns_raw=event_returns_raw,
        contract_event_map=cmap,
        row_counts_raw=counts,
    )


def normalize_contracts(kalshi: pd.DataFrame, polymarket: pd.DataFrame) -> pd.DataFrame:
    """Build a platform-unified contract table."""
    k_rows = []
    for _, r in kalshi.iterrows():
        k_rows.append(
            {
                "platform": "KALSHI",
                "contract_id": r["ticker"],
                "family_id": r.get("event_ticker"),
                "title": r.get("title"),
                "outcome_label": r.get("yes_sub_title"),
                "event_id": r.get("event_id") if pd.notna(r.get("event_id")) else pd.NA,
                "band_id": r.get("band_id") if pd.notna(r.get("band_id")) else pd.NA,
                "open_time": r.get("open_time"),
                "close_time": r.get("close_time"),
                "edge_case_type": r.get("edge_case_type", "none"),
                "volume": r.get("volume"),
                "open_interest": r.get("open_interest"),
            }
        )
    p_rows = []
    for _, r in polymarket.iterrows():
        p_rows.append(
            {
                "platform": "POLYMARKET",
                "contract_id": r["condition_id"],
                "family_id": r.get("slug"),
                "title": r.get("question"),
                "outcome_label": r.get("outcomes"),
                "event_id": r.get("event_id") if pd.notna(r.get("event_id")) else pd.NA,
                "band_id": r.get("band_id") if pd.notna(r.get("band_id")) else pd.NA,
                "open_time": r.get("created_at"),
                "close_time": r.get("end_date"),
                "edge_case_type": r.get("edge_case_type", "none"),
                "volume": r.get("volume"),
                "open_interest": r.get("liquidity"),
            }
        )
    return pd.DataFrame(k_rows + p_rows)
