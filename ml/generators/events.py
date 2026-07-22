"""Event calendar and latent factor generation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ml.synth_config import SynthConfig


REGIME_SCHEDULE = [
    # (end_exclusive, regime)
    ("2022-03-01", "HIGH_INFLATION"),
    ("2023-07-01", "TIGHTENING"),
    ("2024-09-01", "NORMAL"),
    ("2099-01-01", "EASING"),
]

MACRO_VARIABLE = {
    "CPI": "CPI_YoY",
    "FOMC": "PolicyRate",
    "FED_CUTS": "PolicyRate",
    "NFP": "UnemploymentRate",
    "NOISE": "Other",
}


def _regime_for_date(ts: pd.Timestamp) -> str:
    for end, regime in REGIME_SCHEDULE:
        if ts < pd.Timestamp(end, tz="UTC"):
            return regime
    return "NORMAL"


def _year_span(start: pd.Timestamp, end: pd.Timestamp) -> List[int]:
    return list(range(start.year, end.year + 1))


def _cpi_dates(year: int, rng: np.random.Generator) -> List[pd.Timestamp]:
    """Approximate mid-month CPI releases (08:30 ET ~ 13:30 UTC)."""
    dates = []
    for month in range(1, 13):
        day = int(rng.integers(10, 15))
        hour_jitter = int(rng.integers(0, 3))
        ts = pd.Timestamp(
            year=year, month=month, day=min(day, 28), hour=13, minute=30 + hour_jitter, tz="UTC"
        )
        dates.append(ts)
    return dates


def _fomc_dates(year: int, rng: np.random.Generator) -> List[pd.Timestamp]:
    """Eight FOMC-like decision times (14:00 ET ~ 19:00 UTC)."""
    # Rough FOMC months
    months = [1, 3, 5, 6, 7, 9, 11, 12]
    dates = []
    for month in months:
        day = int(rng.choice([15, 16, 17, 18, 19, 20, 21]))
        ts = pd.Timestamp(year=year, month=month, day=min(day, 28), hour=19, minute=0, tz="UTC")
        dates.append(ts)
    return dates


def _nfp_dates(year: int, rng: np.random.Generator) -> List[pd.Timestamp]:
    """First Friday-ish NFP releases."""
    dates = []
    for month in range(1, 13):
        # Pick a day near the first Friday
        day = int(rng.integers(1, 8))
        ts = pd.Timestamp(year=year, month=month, day=day, hour=13, minute=30, tz="UTC")
        # Shift to Friday if needed
        while ts.dayofweek != 4:
            ts += pd.Timedelta(days=1)
            if ts.month != month:
                ts = pd.Timestamp(year=year, month=month, day=1, hour=13, minute=30, tz="UTC")
                break
        dates.append(ts)
    return dates


def generate_events_and_latents(
    cfg: SynthConfig, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate master event calendar and hidden latent factors.

    Returns
    -------
    events : DataFrame
        Public event metadata.
    latents : DataFrame
        Hidden Z_e, V_e, L_e used only for generation.
    """
    start = pd.Timestamp(cfg.start_date, tz="UTC")
    end = pd.Timestamp(cfg.end_date, tz="UTC") + pd.Timedelta(hours=23, minutes=59)

    rows: List[dict] = []
    event_id = 1

    for year in _year_span(start, end):
        for ts in _cpi_dates(year, rng):
            if start <= ts <= end:
                rows.append(_event_row(event_id, "CPI", ts))
                event_id += 1
        for ts in _fomc_dates(year, rng):
            if start <= ts <= end:
                rows.append(_event_row(event_id, "FOMC", ts))
                event_id += 1
        for ts in _nfp_dates(year, rng):
            if start <= ts <= end:
                rows.append(_event_row(event_id, "NFP", ts))
                event_id += 1

        n_noise = int(rng.poisson(cfg.noise_events_per_year))
        for _ in range(n_noise):
            month = int(rng.integers(1, 13))
            day = int(rng.integers(1, 28))
            hour = int(rng.integers(12, 21))
            ts = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0, tz="UTC")
            if start <= ts <= end:
                rows.append(_event_row(event_id, "NOISE", ts))
                event_id += 1

    events = pd.DataFrame(rows).sort_values("release_time").reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1, dtype=int)

    n = len(events)
    z = rng.normal(0.0, cfg.z_std, size=n)
    v = rng.lognormal(cfg.v_lognormal_mean, cfg.v_lognormal_sigma, size=n)
    l = rng.lognormal(cfg.l_lognormal_mean, cfg.l_lognormal_sigma, size=n)

    # Rare FED_CUTS: convert some FOMC in EASING with strongly negative Z
    for i, row in events.iterrows():
        if (
            row["event_type"] == "FOMC"
            and row["regime"] == "EASING"
            and z[i] < -1.25
            and rng.random() < 0.55
        ):
            events.at[i, "event_type"] = "FED_CUTS"
            events.at[i, "macro_variable"] = MACRO_VARIABLE["FED_CUTS"]

    latents = pd.DataFrame(
        {
            "event_id": events["event_id"].values,
            "Z_e": z,
            "V_e": v,
            "L_e": l,
        }
    )
    return events, latents


def _event_row(event_id: int, event_type: str, ts: pd.Timestamp) -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "macro_variable": MACRO_VARIABLE[event_type],
        "release_time": ts.isoformat(),
        "regime": _regime_for_date(ts),
        "day_of_week": ts.day_name(),
    }
