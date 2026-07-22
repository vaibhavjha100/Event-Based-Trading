"""Daily VIX series generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.synth_config import SynthConfig


def generate_vix_daily(
    events: pd.DataFrame,
    latents: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a daily VIX path over the configured horizon.

    VIX follows a mean-reverting process and spikes on large-|Z_e| event days.
    """
    dates = pd.bdate_range(cfg.start_date, cfg.end_date, freq="C")
    n = len(dates)
    level = 18.0
    path = np.empty(n)
    kappa = 0.08
    mu = 18.0
    sigma = 1.1

    event_day_shock = {}
    merged = events.merge(latents, on="event_id")
    for _, row in merged.iterrows():
        d = pd.Timestamp(row["release_time"]).tz_localize(None).normalize()
        shock = 2.5 * abs(float(row["Z_e"])) * np.sqrt(float(row["V_e"]))
        event_day_shock[d] = event_day_shock.get(d, 0.0) + shock

    for i, d in enumerate(dates):
        level = level + kappa * (mu - level) + rng.normal(0.0, sigma)
        level += event_day_shock.get(d.normalize(), 0.0)
        level = float(np.clip(level, 9.0, 80.0))
        path[i] = level

    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "vix_close": np.round(path, 4)})
