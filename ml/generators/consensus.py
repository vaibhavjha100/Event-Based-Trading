"""Consensus forecasts and Surprise(e) generation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ml.synth_config import SynthConfig

# Baseline levels by macro variable / regime
BASELINES: Dict[Tuple[str, str], float] = {
    ("CPI_YoY", "HIGH_INFLATION"): 6.5,
    ("CPI_YoY", "TIGHTENING"): 4.2,
    ("CPI_YoY", "NORMAL"): 2.8,
    ("CPI_YoY", "EASING"): 2.4,
    ("PolicyRate", "HIGH_INFLATION"): 0.5,
    ("PolicyRate", "TIGHTENING"): 4.5,
    ("PolicyRate", "NORMAL"): 5.25,
    ("PolicyRate", "EASING"): 4.0,
    ("UnemploymentRate", "HIGH_INFLATION"): 4.0,
    ("UnemploymentRate", "TIGHTENING"): 3.7,
    ("UnemploymentRate", "NORMAL"): 4.1,
    ("UnemploymentRate", "EASING"): 4.5,
    ("Other", "HIGH_INFLATION"): 50.0,
    ("Other", "TIGHTENING"): 50.0,
    ("Other", "NORMAL"): 50.0,
    ("Other", "EASING"): 50.0,
}

ALPHA_Z: Dict[str, float] = {
    "CPI_YoY": 0.55,
    "PolicyRate": 0.20,
    "UnemploymentRate": 0.15,
    "Other": 1.0,
}


def generate_consensus(
    events: pd.DataFrame,
    latents: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate consensus forecasts, Actual(e), and Surprise(e).

    Surprise is correlated with Z_e but imperfect:
    Surprise ≈ surprise_beta * scaled(Z_e) + noise.
    """
    merged = events.merge(latents, on="event_id")
    rows: List[dict] = []

    for _, row in merged.iterrows():
        macro = row["macro_variable"]
        regime = row["regime"]
        z = float(row["Z_e"])
        baseline = BASELINES.get((macro, regime), 50.0)
        alpha = ALPHA_Z.get(macro, 0.5)

        actual = baseline + alpha * z + rng.normal(0.0, cfg.actual_noise_std)
        # Consensus under-reacts to Z so surprise retains correlation with Z
        consensus = (
            actual
            - cfg.surprise_beta * alpha * z
            + rng.normal(0.0, cfg.consensus_noise_std)
        )
        surprise = actual - consensus

        release = pd.Timestamp(row["release_time"])
        # One or two consensus updates before release
        n_updates = int(rng.integers(1, 3))
        offsets_days = [7, 1][:n_updates]
        for d in offsets_days:
            # Mild drift of consensus toward actual as release approaches
            drift = rng.normal(0.0, 0.05) * (1.0 / max(d, 1))
            c_val = consensus + drift
            rows.append(
                {
                    "event_id": int(row["event_id"]),
                    "macro_variable": macro,
                    "consensus_value": float(c_val),
                    "consensus_time": (release - pd.Timedelta(days=d)).isoformat(),
                    "surprise_value": float(actual - c_val),
                    "actual_value": float(actual),
                }
            )

    return pd.DataFrame(rows)
