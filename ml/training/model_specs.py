"""Estimator builders for OLS, Elastic Net, and GBDT."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline


def make_imputer_pipeline(estimator) -> Pipeline:
    """Median imputer + estimator (imputer fit on train fold/matrix only)."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def build_ols() -> Pipeline:
    return make_imputer_pipeline(LinearRegression())


def build_elasticnet(alpha: float = 0.01, l1_ratio: float = 0.5, max_iter: int = 5000) -> Pipeline:
    return make_imputer_pipeline(
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=42)
    )


def build_gbdt(params: Optional[dict] = None, random_state: int = 42) -> Tuple[Pipeline, str]:
    """Return (pipeline, backend_name). Prefer LightGBM; fallback HistGradientBoosting."""
    params = dict(params or {})
    try:
        from lightgbm import LGBMRegressor

        defaults = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "num_leaves": 15,
            "random_state": random_state,
            "verbosity": -1,
        }
        defaults.update(params)
        est = LGBMRegressor(**defaults)
        return make_imputer_pipeline(est), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        est = HistGradientBoostingRegressor(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=params.get("n_estimators", 100),
            random_state=random_state,
        )
        return make_imputer_pipeline(est), "hist_gradient_boosting"


def expanding_window_splits(n: int, n_folds: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Chronological expanding-window splits (no shuffle).

    Fold k: train on [0, train_end), validate on [train_end, val_end).
    """
    if n < n_folds + 2:
        # Too small: single holdout last 20%
        split = max(1, int(n * 0.8))
        if split >= n:
            split = n - 1
        return [(np.arange(0, split), np.arange(split, n))]

    # Reserve last ~25% as final validation blocks progressively
    min_train = max(n // (n_folds + 1), 5)
    folds = []
    for i in range(1, n_folds + 1):
        train_end = min_train + int((n - min_train) * (i / (n_folds + 1)))
        val_end = min_train + int((n - min_train) * ((i + 1) / (n_folds + 1)))
        if val_end <= train_end:
            val_end = min(train_end + max(1, n // 10), n)
        if train_end < 2 or val_end <= train_end:
            continue
        folds.append((np.arange(0, train_end), np.arange(train_end, val_end)))
    if not folds:
        split = max(1, int(n * 0.8))
        folds = [(np.arange(0, split), np.arange(split, n))]
    return folds


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))
