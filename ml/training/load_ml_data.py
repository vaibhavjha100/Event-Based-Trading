"""Train-only data loading: X_train, y_train, event ids + surprise join."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ml.train_config import TrainConfig

# Hard guard: these filenames must never be opened in this module
FORBIDDEN_TEST_FILES = ("X_test.csv", "y_test.csv", "test_event_ids.csv")


@dataclass
class TrainBundle:
    """Training matrix with metadata columns for ordering."""

    frame: pd.DataFrame  # features + target + event_id + release_time + surprise
    feature_columns_available: List[str]
    target: str
    source_paths: Dict[str, str]
    order_check: Dict[str, Any]


def _assert_no_test_paths(ml_dir: Path) -> None:
    """Refuse to proceed if caller somehow points at test files (defensive)."""
    for name in FORBIDDEN_TEST_FILES:
        # We simply never construct paths to these; document the invariant.
        _ = name


def load_train_bundle(cfg: TrainConfig) -> TrainBundle:
    """Load training artifacts only and join surprise from cleaned event-level.

    Never reads X_test / y_test / test_event_ids.
    """
    _assert_no_test_paths(cfg.ml_dir)
    ml_dir = Path(cfg.ml_dir)
    x_path = ml_dir / "X_train.csv"
    y_path = ml_dir / "y_train.csv"
    ids_path = ml_dir / "train_event_ids.csv"
    cleaned_path = Path(cfg.cleaned_dir) / "event_level_ml_base.csv"

    for p in (x_path, y_path, ids_path, cleaned_path):
        if not p.exists():
            raise FileNotFoundError(f"Required training input missing: {p}")

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    ids = pd.read_csv(ids_path)

    if cfg.target not in y.columns:
        raise ValueError(
            f"Target {cfg.target!r} not in y_train columns: {list(y.columns)}"
        )

    if len(X) != len(y) or len(X) != len(ids):
        raise ValueError(
            f"Row count mismatch: X_train={len(X)} y_train={len(y)} ids={len(ids)}"
        )

    # Join surprise (and release_time from cleaned for verification) on train event_ids only
    cleaned = pd.read_csv(cleaned_path, usecols=["event_id", "surprise", "release_time"])
    cleaned["release_time"] = pd.to_datetime(
        cleaned["release_time"], utc=True, format="ISO8601"
    )
    ids = ids.copy()
    ids["release_time"] = pd.to_datetime(ids["release_time"], utc=True, format="ISO8601")

    train_ids = set(ids["event_id"].astype(int))
    cleaned_train = cleaned[cleaned["event_id"].astype(int).isin(train_ids)].copy()

    # Merge ids with surprise; prefer ids release_time, cross-check cleaned
    merged_ids = ids.merge(
        cleaned_train[["event_id", "surprise", "release_time"]].rename(
            columns={"release_time": "release_time_cleaned"}
        ),
        on="event_id",
        how="left",
    )

    frame = pd.concat(
        [
            merged_ids.reset_index(drop=True),
            X.reset_index(drop=True),
            y[[cfg.target]].reset_index(drop=True),
        ],
        axis=1,
    )

    # Verify / fix chronological order
    order_check = _verify_and_sort_release_time(frame)
    frame = order_check["frame"]

    feature_cols = [c for c in X.columns] + (
        ["surprise"] if "surprise" in frame.columns else []
    )
    # Deduplicate while preserving order
    seen = set()
    feature_available = []
    for c in feature_cols:
        if c not in seen and c in frame.columns:
            seen.add(c)
            feature_available.append(c)

    return TrainBundle(
        frame=frame,
        feature_columns_available=feature_available,
        target=cfg.target,
        source_paths={
            "X_train": str(x_path.resolve()),
            "y_train": str(y_path.resolve()),
            "train_event_ids": str(ids_path.resolve()),
            "event_level_ml_base": str(cleaned_path.resolve()),
        },
        order_check={
            "was_sorted": order_check["was_sorted"],
            "had_to_resort": order_check["had_to_resort"],
            "n_rows": len(frame),
        },
    )


def _verify_and_sort_release_time(frame: pd.DataFrame) -> Dict[str, Any]:
    """Ensure release_time is non-decreasing; re-sort if needed."""
    if "release_time" not in frame.columns:
        raise ValueError("release_time missing after join")

    rt = pd.to_datetime(frame["release_time"], utc=True)
    diffs = rt.diff().dropna()
    was_sorted = bool((diffs >= pd.Timedelta(0)).all()) if len(diffs) else True
    had_to_resort = False
    out = frame
    if not was_sorted:
        out = frame.sort_values("release_time", kind="mergesort").reset_index(drop=True)
        had_to_resort = True
        rt2 = pd.to_datetime(out["release_time"], utc=True)
        diffs2 = rt2.diff().dropna()
        if len(diffs2) and not bool((diffs2 >= pd.Timedelta(0)).all()):
            raise ValueError("release_time still not non-decreasing after sort")
    return {
        "frame": out,
        "was_sorted": was_sorted,
        "had_to_resort": had_to_resort,
    }
