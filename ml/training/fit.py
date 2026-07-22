"""Fit M0–M3 x estimator pipelines on train-only data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.train_config import (
    ELASTICNET_ALPHAS,
    ELASTICNET_L1_RATIOS,
    LGBM_PARAM_GRID,
    TrainConfig,
)
from ml.training.feature_sets import assert_no_leakage
from ml.training.model_specs import (
    build_elasticnet,
    build_gbdt,
    build_ols,
    expanding_window_splits,
    mse,
)


def filter_config_rows(
    frame: pd.DataFrame, feature_names: List[str], target: str
) -> pd.DataFrame:
    """Keep rows with non-null target and at least one non-null feature in the set."""
    if target not in frame.columns:
        raise ValueError(f"Target {target!r} missing")
    sub = frame[feature_names + [target]].copy()
    has_target = sub[target].notna()
    has_any_feat = sub[feature_names].notna().any(axis=1)
    return frame.loc[has_target & has_any_feat].copy()


def fit_all_models(
    frame: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    target: str,
    cfg: TrainConfig,
) -> Dict[str, Dict[str, Any]]:
    """Fit all config × estimator combinations. Returns results keyed by model_id."""
    results: Dict[str, Dict[str, Any]] = {}
    for config_id, features in feature_sets.items():
        assert_no_leakage(features)
        filtered = filter_config_rows(frame, features, target)
        n_rows = len(filtered)
        if n_rows < 3:
            for est in ("ols", "elasticnet", "gbdt"):
                mid = f"{config_id}_{est}"
                results[mid] = _skip(config_id, est, features, target, n_rows, "too_few_rows")
            continue

        X = filtered[features]
        y = filtered[target].astype(float).values

        # OLS
        results[f"{config_id}_ols"] = _fit_ols(config_id, features, target, X, y, n_rows)

        # Elastic Net
        results[f"{config_id}_elasticnet"] = _fit_elasticnet(
            config_id, features, target, X, y, n_rows, cfg
        )

        # GBDT
        results[f"{config_id}_gbdt"] = _fit_gbdt(
            config_id, features, target, X, y, n_rows, cfg
        )

    return results


def _skip(
    config_id: str,
    estimator_name: str,
    feature_names: List[str],
    target: str,
    n_rows: int,
    reason: str,
) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "config_id": config_id,
        "estimator_name": estimator_name,
        "feature_names": list(feature_names),
        "target": target,
        "n_rows_used": n_rows,
        "reason": reason,
        "artifact": None,
    }


def _ok_artifact(
    config_id: str,
    estimator_name: str,
    feature_names: List[str],
    target: str,
    n_rows: int,
    pipeline,
    extra: Optional[dict] = None,
) -> Dict[str, Any]:
    art = {
        "pipeline": pipeline,
        "feature_names": list(feature_names),
        "config_id": config_id,
        "estimator_name": estimator_name,
        "target": target,
        "n_rows_used": n_rows,
    }
    if extra:
        art.update(extra)
    return {
        "status": "trained",
        "config_id": config_id,
        "estimator_name": estimator_name,
        "feature_names": list(feature_names),
        "target": target,
        "n_rows_used": n_rows,
        "reason": None,
        "artifact": art,
        "extra": extra or {},
    }


def _fit_ols(config_id, features, target, X, y, n_rows) -> Dict[str, Any]:
    try:
        pipe = build_ols()
        pipe.fit(X, y)
        coefs = _linear_coefs(pipe, features)
        return _ok_artifact(config_id, "ols", features, target, n_rows, pipe, {"coefficients": coefs})
    except Exception as exc:  # noqa: BLE001
        return _skip(config_id, "ols", features, target, n_rows, f"fit_error:{exc}")


def _fit_elasticnet(config_id, features, target, X, y, n_rows, cfg: TrainConfig) -> Dict[str, Any]:
    try:
        best_alpha, best_l1, best_score = _tune_elasticnet(X, y, cfg)
        pipe = build_elasticnet(alpha=best_alpha, l1_ratio=best_l1)
        pipe.fit(X, y)
        coefs = _linear_coefs(pipe, features)
        return _ok_artifact(
            config_id,
            "elasticnet",
            features,
            target,
            n_rows,
            pipe,
            {
                "coefficients": coefs,
                "best_alpha": best_alpha,
                "best_l1_ratio": best_l1,
                "cv_mse": best_score,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _skip(config_id, "elasticnet", features, target, n_rows, f"fit_error:{exc}")


def _fit_gbdt(config_id, features, target, X, y, n_rows, cfg: TrainConfig) -> Dict[str, Any]:
    try:
        best_params, best_score, backend = _tune_gbdt(X, y, cfg)
        pipe, backend2 = build_gbdt(best_params, random_state=cfg.random_state)
        pipe.fit(X, y)
        return _ok_artifact(
            config_id,
            "gbdt",
            features,
            target,
            n_rows,
            pipe,
            {"best_params": best_params, "cv_mse": best_score, "backend": backend2},
        )
    except Exception as exc:  # noqa: BLE001
        return _skip(config_id, "gbdt", features, target, n_rows, f"fit_error:{exc}")


def _tune_elasticnet(X: pd.DataFrame, y: np.ndarray, cfg: TrainConfig) -> Tuple[float, float, float]:
    folds = expanding_window_splits(len(X), cfg.n_expanding_folds)
    best = (ELASTICNET_ALPHAS[0], ELASTICNET_L1_RATIOS[0], float("inf"))
    for alpha in ELASTICNET_ALPHAS:
        for l1 in ELASTICNET_L1_RATIOS:
            scores = []
            for tr_idx, va_idx in folds:
                pipe = build_elasticnet(alpha=alpha, l1_ratio=l1)
                pipe.fit(X.iloc[tr_idx], y[tr_idx])
                pred = pipe.predict(X.iloc[va_idx])
                scores.append(mse(y[va_idx], pred))
            mean_score = float(np.mean(scores))
            if mean_score < best[2]:
                best = (alpha, l1, mean_score)
    return best


def _tune_gbdt(X: pd.DataFrame, y: np.ndarray, cfg: TrainConfig) -> Tuple[dict, float, str]:
    folds = expanding_window_splits(len(X), cfg.n_expanding_folds)
    best_params = dict(LGBM_PARAM_GRID[0])
    best_score = float("inf")
    backend = "lightgbm"
    for params in LGBM_PARAM_GRID:
        scores = []
        try:
            for tr_idx, va_idx in folds:
                pipe, backend = build_gbdt(params, random_state=cfg.random_state)
                pipe.fit(X.iloc[tr_idx], y[tr_idx])
                pred = pipe.predict(X.iloc[va_idx])
                scores.append(mse(y[va_idx], pred))
        except Exception:
            continue
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        if mean_score < best_score:
            best_score = mean_score
            best_params = dict(params)
    if best_score == float("inf"):
        pipe, backend = build_gbdt(best_params, random_state=cfg.random_state)
        pipe.fit(X, y)
        return best_params, float("nan"), backend
    return best_params, best_score, backend


def _linear_coefs(pipeline, feature_names: List[str]) -> Dict[str, float]:
    model = pipeline.named_steps["model"]
    coef = getattr(model, "coef_", None)
    if coef is None:
        return {}
    coef = np.asarray(coef).ravel()
    out = {f: float(c) for f, c in zip(feature_names, coef)}
    intercept = getattr(model, "intercept_", None)
    if intercept is not None:
        out["__intercept__"] = float(np.asarray(intercept).ravel()[0])
    return out
