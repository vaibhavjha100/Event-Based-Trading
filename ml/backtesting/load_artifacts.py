"""Load trained model pickles and ML train/test frames."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml.backtest_config import (
    CONFIG_IDS,
    ESTIMATORS,
    EVENT_LEVEL_PATH,
    LEAKAGE_EXACT,
    LEAKAGE_PREFIXES,
    ML_DIR,
    MODELS_DIR,
    PRIMARY_TARGET,
    SPLIT_METADATA_PATH,
)


@dataclass
class BacktestData:
    X_train: pd.DataFrame
    y_train: pd.Series
    train_event_ids: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    test_event_ids: pd.Series
    train_release_time: pd.Series
    test_release_time: pd.Series
    surprise_by_event: pd.Series
    split_meta: Dict[str, Any]
    skip_log: List[Dict[str, str]] = field(default_factory=list)


def _is_leakage(name: str) -> bool:
    if name in LEAKAGE_EXACT:
        return True
    return any(name.startswith(p) for p in LEAKAGE_PREFIXES)


def load_split_frames() -> BacktestData:
    X_train = pd.read_csv(ML_DIR / "X_train.csv")
    X_test = pd.read_csv(ML_DIR / "X_test.csv")
    y_train_df = pd.read_csv(ML_DIR / "y_train.csv")
    y_test_df = pd.read_csv(ML_DIR / "y_test.csv")
    train_ids = pd.read_csv(ML_DIR / "train_event_ids.csv")["event_id"]
    test_ids = pd.read_csv(ML_DIR / "test_event_ids.csv")["event_id"]

    y_train = y_train_df[PRIMARY_TARGET]
    y_test = y_test_df[PRIMARY_TARGET]

    event = pd.read_csv(EVENT_LEVEL_PATH)
    event["release_time"] = pd.to_datetime(event["release_time"], utc=True)
    surprise = event.set_index("event_id")["surprise"]
    release = event.set_index("event_id")["release_time"]

    train_release = train_ids.map(release)
    test_release = test_ids.map(release)

    with open(SPLIT_METADATA_PATH, encoding="utf-8") as f:
        split_meta = json.load(f)

    return BacktestData(
        X_train=X_train,
        y_train=y_train,
        train_event_ids=train_ids,
        X_test=X_test,
        y_test=y_test,
        test_event_ids=test_ids,
        train_release_time=train_release,
        test_release_time=test_release,
        surprise_by_event=surprise,
        split_meta=split_meta,
    )


def load_artifacts(
    models_dir: Path = MODELS_DIR,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
    """Load all M0–M3 estimator pickles; skip missing with reason."""
    artifacts: Dict[str, Dict[str, Any]] = {}
    skips: List[Dict[str, str]] = []

    for cfg in CONFIG_IDS:
        for est in ESTIMATORS:
            model_id = f"{cfg}_{est}"
            path = models_dir / f"{model_id}.pkl"
            if not path.exists():
                skips.append({"model_id": model_id, "reason": f"missing file: {path.name}"})
                continue
            try:
                with open(path, "rb") as f:
                    art = pickle.load(f)
            except Exception as e:
                skips.append({"model_id": model_id, "reason": f"load_error: {e}"})
                continue

            feats = list(art.get("feature_names") or [])
            leak = [c for c in feats if _is_leakage(c)]
            if leak:
                skips.append(
                    {
                        "model_id": model_id,
                        "reason": f"leakage features in artifact: {leak}",
                    }
                )
                continue
            if art.get("target") and art["target"] != PRIMARY_TARGET:
                # Still allow if target matches primary; warn otherwise
                if PRIMARY_TARGET not in str(art.get("target")):
                    skips.append(
                        {
                            "model_id": model_id,
                            "reason": f"target mismatch: {art.get('target')}",
                        }
                    )
                    continue

            art["model_id"] = model_id
            art["path"] = str(path)
            artifacts[model_id] = art

    return artifacts, skips
