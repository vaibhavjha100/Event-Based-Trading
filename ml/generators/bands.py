"""Band definitions and platform-level probability distributions."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ml.synth_config import BAND_TIME_OFFSETS, SynthConfig


def _cpi_bands(consensus: float) -> List[Tuple[int, float, str]]:
    """Return (band_id, threshold, label) for CPI YoY bins."""
    center = round(consensus * 2) / 2  # nearest 0.5
    thresholds = [center - 1.0, center - 0.5, center, center + 0.5, center + 1.0, center + 1.5]
    bands = []
    for i, t in enumerate(thresholds):
        if i == 0:
            label = f"Below {t:.1f}%"
        else:
            prev = thresholds[i - 1]
            label = f"{prev:.1f}% to {t:.1f}%"
        bands.append((i, float(t), label))
    # Top open band
    last = thresholds[-1]
    bands.append((len(thresholds), float(last + 0.5), f"Above {last:.1f}%"))
    return bands


def _rate_bands(consensus: float) -> List[Tuple[int, float, str]]:
    center = round(consensus * 4) / 4
    thresholds = [center - 0.50, center - 0.25, center, center + 0.25, center + 0.50]
    bands = []
    for i, t in enumerate(thresholds):
        if i == 0:
            label = f"Below {t:.2f}%"
        else:
            label = f"{thresholds[i - 1]:.2f}% to {t:.2f}%"
        bands.append((i, float(t), label))
    bands.append((len(thresholds), float(thresholds[-1] + 0.25), f"Above {thresholds[-1]:.2f}%"))
    return bands


def _nfp_bands(consensus: float) -> List[Tuple[int, float, str]]:
    center = round(consensus * 10) / 10
    thresholds = [center - 0.3, center - 0.1, center, center + 0.1, center + 0.3]
    bands = []
    for i, t in enumerate(thresholds):
        if i == 0:
            label = f"Below {t:.1f}%"
        else:
            label = f"{thresholds[i - 1]:.1f}% to {t:.1f}%"
        bands.append((i, float(t), label))
    bands.append((len(thresholds), float(thresholds[-1] + 0.1), f"Above {thresholds[-1]:.1f}%"))
    return bands


def bands_for_event(macro_variable: str, consensus: float) -> List[Tuple[int, float, str]]:
    """Define discrete outcome bands for a macro event."""
    if macro_variable == "CPI_YoY":
        return _cpi_bands(consensus)
    if macro_variable == "PolicyRate":
        return _rate_bands(consensus)
    if macro_variable == "UnemploymentRate":
        return _nfp_bands(consensus)
    # Generic noise
    return [(0, 0.0, "Yes"), (1, 1.0, "No")]


def _adverse_mask(
    bands: Sequence[Tuple[int, float, str]], consensus: float, macro_variable: str
) -> List[bool]:
    """Mark hotter/hawkish (or worse-than-consensus) bands as adverse."""
    adverse = []
    for band_id, value, label in bands:
        if macro_variable in ("CPI_YoY", "PolicyRate", "UnemploymentRate"):
            # Higher outcomes than consensus are adverse for equities
            adverse.append(value > consensus + 1e-9 or "Above" in label)
        else:
            adverse.append(band_id == 0)
    # Ensure at least one adverse and one non-adverse
    if all(adverse) or not any(adverse):
        mid = len(bands) // 2
        adverse = [i >= mid for i in range(len(bands))]
    return adverse


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / e.sum()


def _band_logits(
    bands: Sequence[Tuple[int, float, str]],
    adverse: Sequence[bool],
    z_plat: float,
    minutes_to_release: int,
    v_e: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build logits tilted by platform view of the shock."""
    n = len(bands)
    # Neutral peaked around middle
    center = (n - 1) / 2.0
    logits = -0.35 * ((np.arange(n) - center) ** 2)

    # Information arrival: tilt strengthens closer to release (and jumps after)
    if minutes_to_release >= 0:
        info = 1.4
    else:
        info = 0.55 + 0.45 * (1.0 - min(abs(minutes_to_release), 60) / 60.0)

    tilt = z_plat * info
    for i, is_adv in enumerate(adverse):
        logits[i] += tilt if is_adv else -tilt

    logits += rng.normal(0.0, 0.08 * np.sqrt(v_e), size=n)
    return logits


def generate_band_probabilities(
    events: pd.DataFrame,
    latents: pd.DataFrame,
    consensus: pd.DataFrame,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate band-level probabilities for Kalshi and Polymarket at key times."""
    # Latest consensus per event
    cons_latest = (
        consensus.sort_values("consensus_time")
        .groupby("event_id", as_index=False)
        .tail(1)[["event_id", "consensus_value", "actual_value"]]
    )
    merged = events.merge(latents, on="event_id").merge(cons_latest, on="event_id")

    rows: List[dict] = []
    for _, row in merged.iterrows():
        if row["event_type"] == "NOISE":
            continue

        consensus_val = float(row["consensus_value"])
        bands = bands_for_event(row["macro_variable"], consensus_val)
        adverse = _adverse_mask(bands, consensus_val, row["macro_variable"])
        z = float(row["Z_e"])
        v = float(row["V_e"])
        release = pd.Timestamp(row["release_time"])

        z_k = z + rng.normal(0.0, cfg.platform_eps_std)
        z_p = z + rng.normal(0.0, cfg.platform_eps_std)

        for offset in BAND_TIME_OFFSETS:
            ts = release + pd.Timedelta(minutes=offset)
            for platform, z_plat in (("KALSHI", z_k), ("POLYMARKET", z_p)):
                logits = _band_logits(bands, adverse, z_plat, offset, v, rng)
                # Post-release: pull toward actual outcome
                if offset >= 0:
                    actual = float(row["actual_value"])
                    distances = np.array([abs(b[1] - actual) for b in bands], dtype=float)
                    winner = int(np.argmin(distances))
                    logits = logits * 0.3
                    logits[winner] += 2.5
                probs = _softmax(logits)
                for (band_id, value, label), is_adv, p in zip(bands, adverse, probs):
                    rows.append(
                        {
                            "event_id": int(row["event_id"]),
                            "platform": platform,
                            "band_id": int(band_id),
                            "band_value": float(value),
                            "band_label": label,
                            "is_adverse": bool(is_adv),
                            "timestamp": ts.isoformat(),
                            "minutes_to_release": int(offset),
                            "probability": float(p),
                        }
                    )

    return pd.DataFrame(rows)


def get_event_band_defs(
    events: pd.DataFrame, consensus: pd.DataFrame
) -> Dict[int, List[Tuple[int, float, str, bool]]]:
    """Return band definitions with adverse flags keyed by event_id."""
    cons_latest = (
        consensus.sort_values("consensus_time")
        .groupby("event_id", as_index=False)
        .tail(1)[["event_id", "consensus_value"]]
    )
    merged = events.merge(cons_latest, on="event_id", how="left")
    out: Dict[int, List[Tuple[int, float, str, bool]]] = {}
    for _, row in merged.iterrows():
        eid = int(row["event_id"])
        if row["event_type"] == "NOISE" or pd.isna(row.get("consensus_value")):
            out[eid] = [(0, 0.0, "Yes", True), (1, 1.0, "No", False)]
            continue
        c = float(row["consensus_value"])
        bands = bands_for_event(row["macro_variable"], c)
        adverse = _adverse_mask(bands, c, row["macro_variable"])
        out[eid] = [
            (bid, val, lab, adv) for (bid, val, lab), adv in zip(bands, adverse)
        ]
    return out
