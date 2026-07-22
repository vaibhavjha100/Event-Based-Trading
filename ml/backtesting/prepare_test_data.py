"""Align per-model feature matrices and join surprise."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ml.backtesting.load_artifacts import BacktestData


def build_model_matrices(
    data: BacktestData,
    artifact: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train_m, X_test_m, y_train, y_test with artifact feature order.

    Joins `surprise` from event-level onto both train and test by event_id.
    """
    feats = list(artifact["feature_names"])
    Xtr = data.X_train.copy()
    Xte = data.X_test.copy()

    if "surprise" in feats:
        Xtr["surprise"] = data.train_event_ids.map(data.surprise_by_event).astype(float).values
        Xte["surprise"] = data.test_event_ids.map(data.surprise_by_event).astype(float).values

    missing = [c for c in feats if c not in Xtr.columns]
    if missing:
        raise KeyError(f"{artifact.get('model_id')}: missing features {missing}")

    Xtr_m = Xtr[feats].copy()
    Xte_m = Xte[feats].copy()
    return Xtr_m, Xte_m, data.y_train.copy(), data.y_test.copy()
