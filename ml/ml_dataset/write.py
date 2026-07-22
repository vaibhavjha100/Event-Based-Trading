"""Write ML-ready artifacts under data/ml/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ml.ml_dataset.split import SplitResult


def write_ml_artifacts(
    *,
    out_dir: Path,
    split: SplitResult,
    feature_columns: list[str],
    feature_groups: pd.DataFrame,
    target_columns: list[str],
    feature_meta: Dict[str, Any],
    source_path: str,
    universe_rule: str,
    dropped_noise_rows: int,
    n_before_core_filter: int,
) -> Dict[str, Any]:
    """Write train/test CSVs, column lists, IDs, full table, and metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = split.train[feature_columns]
    X_test = split.test[feature_columns]
    y_train = split.train[target_columns]
    y_test = split.test[target_columns]

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    feature_groups.to_csv(out_dir / "feature_columns.csv", index=False)
    pd.DataFrame({"target": target_columns}).to_csv(
        out_dir / "target_columns.csv", index=False
    )

    id_cols = [c for c in ("event_id", "release_time") if c in split.train.columns]
    split.train[id_cols].to_csv(out_dir / "train_event_ids.csv", index=False)
    split.test[id_cols].to_csv(out_dir / "test_event_ids.csv", index=False)

    full_cols = id_cols + feature_columns + target_columns
    # preserve order, unique
    seen = set()
    ordered = []
    for c in full_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    full = pd.concat([split.train[ordered], split.test[ordered]], ignore_index=True)
    full.to_csv(out_dir / "ml_dataset_full.csv", index=False)

    meta: Dict[str, Any] = {
        "source_path": source_path,
        "universe_rule": universe_rule,
        "dropped_noise_rows": dropped_noise_rows,
        "n_after_universe_sort": n_before_core_filter,
        "n_after_core_target_filter": split.n_train + split.n_test,
        "n_train": split.n_train,
        "n_test": split.n_test,
        "train_ratio": split.train_ratio,
        "split_type": "chronological",
        "shuffled": False,
        "split_boundary_release_time": split.split_boundary_release_time,
        "core_features_used": feature_meta["core_features_used"],
        "optional_features_used": feature_meta["optional_features_used"],
        "optional_features_absent": feature_meta["optional_features_absent"],
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "excluded_leakage_columns": feature_meta["excluded_leakage_columns"],
        "files": [
            "X_train.csv",
            "X_test.csv",
            "y_train.csv",
            "y_test.csv",
            "feature_columns.csv",
            "target_columns.csv",
            "train_event_ids.csv",
            "test_event_ids.csv",
            "ml_dataset_full.csv",
            "split_metadata.json",
        ],
    }
    (out_dir / "split_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return meta
