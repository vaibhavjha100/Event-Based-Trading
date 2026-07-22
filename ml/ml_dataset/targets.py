"""Target column resolution and matrix construction."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from ml.ml_data_config import MLDataConfig


def resolve_target_columns(df: pd.DataFrame, cfg: MLDataConfig) -> List[str]:
    """Return ordered target columns to export."""
    targets = list(cfg.primary_targets)
    if cfg.include_additional_targets:
        for c in cfg.additional_targets:
            if c in df.columns and c not in targets:
                targets.append(c)

    missing = [c for c in cfg.primary_targets if c not in df.columns]
    if missing:
        raise ValueError("Missing primary target columns: " + ", ".join(missing))
    return targets


def build_target_matrix(df: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    """Build y matrix."""
    return df[target_columns].copy()


def target_columns_table(target_columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"target": target_columns})
