"""Website handoff: model_bundle, metadata, simulated features, fixture."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.backtest_config import (
    N_FIXTURE_ROWS,
    N_SIMULATED_ROWS,
    PACKAGE_VERSION_KEYS,
    PRIMARY_TARGET,
    SIM_SEED,
    TAU_FLOOR,
    TAU_STD_FRAC,
    WALKFORWARD_MODE,
)
from ml.backtesting.io_utils import package_versions, write_csv, write_json
from ml.backtesting.select_best_model import SELECTION_RULE_TEXT
from ml.backtesting.signal_policy import apply_signals


class SklearnReturnModel:
    """Thin wrapper: .predict(X) in feature_columns order; preprocessing in pipeline."""

    def __init__(self, pipeline, feature_columns: List[str]):
        self.pipeline = pipeline
        self.feature_columns = list(feature_columns)

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_columns if c not in X.columns]
            if missing:
                raise KeyError(f"Missing feature columns: {missing}")
            frame = X[self.feature_columns]
        else:
            mat = np.asarray(X, dtype=float)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            if mat.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"Expected {len(self.feature_columns)} columns, got {mat.shape[1]}"
                )
            frame = pd.DataFrame(mat, columns=self.feature_columns)
        return np.asarray(self.pipeline.predict(frame), dtype=float)


def build_model_bundle(artifact: Dict[str, Any]) -> Dict[str, Any]:
    feats = list(artifact["feature_names"])
    wrapper = SklearnReturnModel(artifact["pipeline"], feats)
    return {
        "artifact": artifact,
        "model": wrapper,
        "feature_columns": feats,
        "preprocessing_included": True,
        "target": artifact.get("target", PRIMARY_TARGET),
        "estimator_name": artifact.get("estimator_name"),
        "config_id": artifact.get("config_id"),
        "model_id": artifact.get("model_id"),
    }


def _feature_ranges(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    ranges: Dict[str, Dict[str, float]] = {}
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        lo = float(np.nanpercentile(s, 5)) if s.notna().any() else 0.0
        hi = float(np.nanpercentile(s, 95)) if s.notna().any() else 1.0
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = lo + 1.0
        if hi <= lo:
            hi = lo + 1e-3
        ranges[c] = {"lo": lo, "hi": hi, "median": float(np.nanmedian(s))}
    return ranges


def generate_simulated_features(
    feature_columns: List[str],
    reference_X: pd.DataFrame,
    n_rows: int = N_SIMULATED_ROWS,
    seed: int = SIM_SEED,
) -> pd.DataFrame:
    """Demo-only rows: website schema + every feature_column with plausible values."""
    rng = np.random.default_rng(seed)
    ranges = _feature_ranges(reference_X)

    website_meta = {
        "date": pd.date_range("2025-06-01", periods=n_rows, freq="7D", tz="UTC"),
        "market_title": [f"Demo macro event {i+1}" for i in range(n_rows)],
        "category": rng.choice(["CPI", "NFP", "FOMC", "FED_CUTS"], size=n_rows),
        "target_asset": ["SPY"] * n_rows,
        "p_bad": rng.uniform(0.15, 0.85, size=n_rows),
        "pmps_pre": rng.uniform(-0.15, 0.15, size=n_rows),
        "volume": rng.integers(500, 50000, size=n_rows),
        "open_interest": rng.integers(1000, 100000, size=n_rows),
        "classification_confidence": rng.uniform(0.55, 0.98, size=n_rows),
        "known_at": pd.date_range("2025-06-01", periods=n_rows, freq="7D", tz="UTC")
        - pd.Timedelta(minutes=5),
    }
    rows: Dict[str, Any] = dict(website_meta)

    for c in feature_columns:
        if c.startswith("regime_") or c.startswith("event_type_"):
            # one-hot-ish demo: mostly 0/1
            rows[c] = rng.integers(0, 2, size=n_rows).astype(float)
        elif c == "surprise":
            rows[c] = rng.normal(0, 0.5, size=n_rows)
        elif c in ranges:
            r = ranges[c]
            rows[c] = rng.uniform(r["lo"], r["hi"], size=n_rows)
        else:
            rows[c] = rng.normal(0, 0.1, size=n_rows)

    # Align some website columns with feature names when present
    if "p_bad_Kalshi_t0_5" in feature_columns:
        rows["p_bad"] = rows["p_bad_Kalshi_t0_5"]
    if "PMPS_pre_Kalshi" in feature_columns:
        rows["pmps_pre"] = rows["PMPS_pre_Kalshi"]

    df = pd.DataFrame(rows)
    # Ensure all feature columns present
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0
    return df


def build_prediction_fixture(
    bundle: Dict[str, Any],
    tau: float,
    simulated: pd.DataFrame,
    n_rows: int = N_FIXTURE_ROWS,
) -> pd.DataFrame:
    feats = bundle["feature_columns"]
    model: SklearnReturnModel = bundle["model"]
    sub = simulated.head(n_rows).copy()
    X = sub[feats]
    preds = model.predict(X)
    signals = apply_signals(preds, tau)
    out = X.copy()
    out["expected_prediction"] = preds
    out["expected_signal"] = signals
    return out


def write_website_outputs(
    output_dir: Path,
    selected_artifact: Dict[str, Any],
    tau: float,
    selection_info: Dict[str, Any],
    split_meta: Dict[str, Any],
    reference_X: pd.DataFrame,
    methodology_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write model_bundle.pkl, model_metadata.json, simulated_features, fixture, assumptions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    bundle = build_model_bundle(selected_artifact)
    bundle_path = output_dir / "model_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    paths["model_bundle"] = bundle_path

    feats = bundle["feature_columns"]
    train_end = split_meta.get("split_boundary_release_time")
    meta = {
        "version": "1.0",
        "type": "sklearn_return_model",
        "target": PRIMARY_TARGET,
        "prediction_kind": "continuous_return",
        "feature_columns": feats,
        "thresholds": {
            "tau": tau,
            "tau_rule": f"max({TAU_FLOOR}, {TAU_STD_FRAC} * std(yhat_train))",
            "long": f"yhat >= +tau",
            "short": f"yhat <= -tau",
            "neutral": "otherwise",
        },
        "train_period": {
            "end_before": train_end,
            "n_rows": split_meta.get("n_train"),
        },
        "val_period": {
            "note": "Hyperparameters tuned via expanding-window CV on train only; no separate held-out val set exported.",
        },
        "test_period": {
            "start_at": train_end,
            "n_rows": split_meta.get("n_test"),
            "walkforward_mode": WALKFORWARD_MODE,
            "label": "historical_walkforward_test",
        },
        "preprocessing_included": True,
        "package_versions": package_versions(PACKAGE_VERSION_KEYS),
        "timezone": "UTC",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_config": methodology_config
        or {
            "config_id": selected_artifact.get("config_id"),
            "estimator_name": selected_artifact.get("estimator_name"),
            "entry_exit": "event window aligned to target (t0-5 -> t0+60)",
            "transaction_costs": "none",
        },
        "estimator_name": selected_artifact.get("estimator_name"),
        "config_id": selected_artifact.get("config_id"),
        "model_id": selected_artifact.get("model_id"),
        "walkforward_mode": WALKFORWARD_MODE,
        "selection_reason": selection_info.get(
            "selection_reason", SELECTION_RULE_TEXT
        ),
        "selection_rule": SELECTION_RULE_TEXT,
    }
    meta_path = output_dir / "model_metadata.json"
    write_json(meta, meta_path)
    paths["model_metadata"] = meta_path

    # Reference matrix needs surprise if in features
    ref = reference_X.copy()
    for c in feats:
        if c not in ref.columns:
            ref[c] = np.nan
    sim = generate_simulated_features(feats, ref[feats], N_SIMULATED_ROWS, SIM_SEED)
    sim_path = output_dir / "simulated_features.csv"
    write_csv(sim, sim_path)
    paths["simulated_features"] = sim_path

    assumptions = f"""# Simulation assumptions

## Purpose
`simulated_features.csv` and `prediction_fixture.csv` are **demo-only** rows for website
integration testing. They are **not** historical walk-forward backtest events.

## Generation method
- Fixed seed: `{SIM_SEED}`
- Rows: `{N_SIMULATED_ROWS}` simulated feature rows; fixture uses first `{N_FIXTURE_ROWS}`
- Website columns: date, market_title, category, target_asset, p_bad, pmps_pre, volume,
  open_interest, classification_confidence, known_at
- Model feature columns: sampled uniformly from approx. 5th–95th percentile ranges of the
  training design matrix (one-hot regime/event flags drawn as 0/1; surprise ~ N(0, 0.5))
- `expected_prediction` / `expected_signal` come from `model_bundle.predict` and train-only τ

## Ranges
Feature ranges are derived from the selected model's training feature matrix percentiles.
Website liquidity fields (volume, open_interest) are synthetic integers for UI display only.

## Trading assumptions (backtest, not demo)
- Entry/exit aligned to target definition: t0−5 → t0+60 for `{PRIMARY_TARGET}`
- No transaction costs, no slippage
- Long / Short / Neutral from train-only threshold τ
- Walk-forward mode: `{WALKFORWARD_MODE}`

## Disclaimer
Do not treat simulated rows as live market data or as out-of-sample research results.
Historical research outputs are labeled separately (`event_results.csv`, etc.).
"""
    assump_path = output_dir / "simulation_assumptions.md"
    assump_path.write_text(assumptions, encoding="utf-8")
    paths["simulation_assumptions"] = assump_path

    fixture = build_prediction_fixture(bundle, tau, sim, N_FIXTURE_ROWS)
    fix_path = output_dir / "prediction_fixture.csv"
    write_csv(fixture, fix_path)
    paths["prediction_fixture"] = fix_path

    return paths
