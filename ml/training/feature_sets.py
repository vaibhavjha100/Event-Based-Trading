"""M0–M3 nested feature-set construction."""

from __future__ import annotations

from typing import Dict, List, Sequence

from ml.train_config import (
    CONTROL_FEATURES,
    LIQUIDITY_FEATURES,
    MOMENT_FEATURES,
    PMPS_FEATURES,
)


def _present(candidates: Sequence[str], available: Sequence[str]) -> List[str]:
    avail = set(available)
    return [c for c in candidates if c in avail]


def build_feature_sets(available_columns: Sequence[str]) -> Dict[str, List[str]]:
    """Return ordered feature lists for M0–M3 (nested)."""
    controls = _present(CONTROL_FEATURES, available_columns)
    surprise = _present(("surprise",), available_columns)
    if not surprise:
        raise ValueError(
            "M0 requires 'surprise' after joining cleaned event_level on train ids"
        )
    if not controls:
        raise ValueError("No control features available for M0")

    m0 = surprise + controls
    m1 = m0 + _present(PMPS_FEATURES, available_columns)
    m2 = m1 + _present(MOMENT_FEATURES, available_columns)
    m3 = m2 + _present(LIQUIDITY_FEATURES, available_columns)

    return {"M0": m0, "M1": m1, "M2": m2, "M3": m3}


def assert_no_leakage(feature_names: Sequence[str]) -> None:
    """Raise if leakage-prone names appear in a feature list."""
    bad = []
    for c in feature_names:
        if c.startswith("reaction_") or c.startswith("ret_"):
            bad.append(c)
        if c in ("actual_value", "event_id", "release_time", "consensus_time"):
            bad.append(c)
    if bad:
        raise ValueError(f"Leakage / forbidden columns in features: {bad}")
