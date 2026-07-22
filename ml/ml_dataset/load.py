"""Load cleaned event-level table, filter universe, sort chronologically."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.ml_data_config import MLDataConfig


def load_and_prepare_events(cfg: MLDataConfig) -> pd.DataFrame:
    """Load event-level ML base, drop excluded event types, sort by release_time."""
    path = Path(cfg.event_level_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Event-level ML base not found: {path}. "
            "Run preprocessing first (python -m ml.run_preprocessing)."
        )

    df = pd.read_csv(path)
    if "release_time" not in df.columns:
        raise ValueError("event_level_ml_base.csv missing required column: release_time")
    if "event_type" not in df.columns:
        raise ValueError("event_level_ml_base.csv missing required column: event_type")

    df["release_time"] = pd.to_datetime(df["release_time"], utc=True, format="ISO8601")

    before = len(df)
    df = df[~df["event_type"].isin(cfg.exclude_event_types)].copy()
    dropped_noise = before - len(df)

    df = df.sort_values("release_time", kind="mergesort").reset_index(drop=True)
    df.attrs["dropped_noise_rows"] = dropped_noise
    df.attrs["source_path"] = str(path.resolve())
    return df
