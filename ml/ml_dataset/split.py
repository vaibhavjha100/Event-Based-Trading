"""Chronological train/test split (no shuffle)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class SplitResult:
    """Containers for a chronological split."""

    train: pd.DataFrame
    test: pd.DataFrame
    n_train: int
    n_test: int
    train_ratio: float
    split_boundary_release_time: str


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.75) -> SplitResult:
    """Split a chronologically sorted frame into earliest train_ratio and remainder.

    Assumes ``df`` is already sorted by release_time ascending.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    n = len(df)
    if n < 2:
        raise ValueError(f"Need at least 2 rows to split; got {n}")

    n_train = int(n * train_ratio)  # floor via int()
    if n_train < 1:
        n_train = 1
    n_test = n - n_train
    if n_test < 1:
        raise ValueError(
            f"Test set empty after split (n={n}, train_ratio={train_ratio}). "
            "Lower train_ratio or add more events."
        )

    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    boundary = str(test["release_time"].iloc[0])

    return SplitResult(
        train=train,
        test=test,
        n_train=n_train,
        n_test=n_test,
        train_ratio=train_ratio,
        split_boundary_release_time=boundary,
    )
