"""Feature column resolution and matrix construction."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from ml.ml_data_config import EXPLICIT_LEAKAGE_COLUMNS, MLDataConfig


def resolve_feature_columns(
    df: pd.DataFrame, cfg: MLDataConfig
) -> Tuple[List[str], List[str], Dict]:
    """Return (core_used, optional_used, meta) for available columns.

    Raises if any configured core feature column is missing from the schema.
    """
    missing_core = [c for c in cfg.core_features if c not in df.columns]
    if missing_core:
        raise ValueError(
            "Missing required core feature columns in event-level table: "
            + ", ".join(missing_core)
        )

    core_used = list(cfg.core_features)
    optional_used = [c for c in cfg.optional_features if c in df.columns]
    optional_absent = [c for c in cfg.optional_features if c not in df.columns]

    # Track leakage-named columns present in the table (for metadata)
    present_leakage = []
    for c in df.columns:
        if c.startswith("reaction_") or c.startswith("ret_"):
            present_leakage.append(c)
        elif c in EXPLICIT_LEAKAGE_COLUMNS:
            present_leakage.append(c)
        elif c in (
            "event_id",
            "release_time",
            "consensus_time",
            "event_type",
            "macro_variable",
            "regime",
            "day_of_week",
        ):
            present_leakage.append(c)

    meta = {
        "core_features_used": core_used,
        "optional_features_used": optional_used,
        "optional_features_absent": optional_absent,
        "excluded_leakage_columns": sorted(set(present_leakage)),
        "feature_columns_ordered": core_used + optional_used,
    }
    return core_used, optional_used, meta


def filter_rows_for_core_and_targets(
    df: pd.DataFrame,
    core_features: List[str],
    primary_targets: Sequence[str],
) -> pd.DataFrame:
    """Keep rows with non-null primary targets and non-null core features.

    Optional feature nulls are allowed.
    """
    missing_targets = [c for c in primary_targets if c not in df.columns]
    if missing_targets:
        raise ValueError(
            "Missing primary target columns: " + ", ".join(missing_targets)
        )

    mask = pd.Series(True, index=df.index)
    for c in primary_targets:
        mask &= df[c].notna()
    for c in core_features:
        mask &= df[c].notna()

    return df.loc[mask].reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Build X matrix (no identifiers)."""
    return df[feature_columns].copy()


def feature_columns_table(core_used: List[str], optional_used: List[str]) -> pd.DataFrame:
    """DataFrame listing feature names and group labels."""
    rows = [{"feature": c, "group": "core"} for c in core_used]
    rows += [{"feature": c, "group": "optional"} for c in optional_used]
    return pd.DataFrame(rows)
