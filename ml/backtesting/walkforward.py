"""Clone pipelines and run expanding-window walk-forward scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline


def clone_pipeline_from_artifact(artifact: Dict[str, Any]) -> Pipeline:
    """Fresh imputer+estimator with stored HPs (no re-tuning)."""
    name = artifact["estimator_name"]
    if name == "ols":
        est = LinearRegression()
    elif name == "elasticnet":
        est = ElasticNet(
            alpha=float(artifact.get("best_alpha", 0.01)),
            l1_ratio=float(artifact.get("best_l1_ratio", 0.5)),
            max_iter=5000,
            random_state=42,
        )
    elif name == "gbdt":
        params = dict(artifact.get("best_params") or {})
        backend = artifact.get("backend", "lightgbm")
        if backend == "lightgbm" or "n_estimators" in params:
            try:
                from lightgbm import LGBMRegressor

                defaults = {
                    "n_estimators": 100,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "num_leaves": 15,
                    "random_state": 42,
                    "verbosity": -1,
                }
                defaults.update(params)
                est = LGBMRegressor(**defaults)
            except ImportError:
                from sklearn.ensemble import HistGradientBoostingRegressor

                est = HistGradientBoostingRegressor(
                    max_depth=params.get("max_depth", 3),
                    learning_rate=params.get("learning_rate", 0.05),
                    max_iter=params.get("n_estimators", 100),
                    random_state=42,
                )
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor

            est = HistGradientBoostingRegressor(
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.05),
                max_iter=params.get("n_estimators", 100),
                random_state=42,
            )
    else:
        raise ValueError(f"Unknown estimator_name: {name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", est),
        ]
    )


def walkforward_expanding_refit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact: Dict[str, Any],
) -> np.ndarray:
    """Predict each test row i using only train + test[0..i-1].

    Never includes row i or future rows in the fit.
    """
    n_test = len(X_test)
    preds = np.full(n_test, np.nan, dtype=float)
    cols = list(X_train.columns)
    ytr = y_train.to_numpy(dtype=float)
    yte = y_test.to_numpy(dtype=float)

    for i in range(n_test):
        if i == 0:
            X_fit = X_train
            y_fit = ytr
        else:
            X_fit = pd.concat([X_train, X_test.iloc[:i]], axis=0, ignore_index=True)
            y_fit = np.concatenate([ytr, yte[:i]])

        mask = np.isfinite(y_fit)
        if mask.sum() < 2:
            continue
        pipe = clone_pipeline_from_artifact(artifact)
        pipe.fit(X_fit.iloc[mask][cols], y_fit[mask])
        preds[i] = float(pipe.predict(X_test.iloc[i : i + 1][cols])[0])

    return preds


def fit_full_train_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifact: Dict[str, Any],
) -> np.ndarray:
    """In-sample predictions on full original train (for threshold fitting)."""
    y = y_train.to_numpy(dtype=float)
    mask = np.isfinite(y)
    pipe = clone_pipeline_from_artifact(artifact)
    pipe.fit(X_train.iloc[mask], y[mask])
    return pipe.predict(X_train).astype(float)
