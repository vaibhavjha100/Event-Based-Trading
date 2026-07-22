"""Configuration for train-only model fitting (M0–M3)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ML_DIR = PROJECT_ROOT / "data" / "ml"
DEFAULT_CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_TARGET = "ret_SPY_60m"

CONTROL_FEATURES: Tuple[str, ...] = (
    "VIX_Tminus1",
    "prior_5d_SPY_return",
    "regime_HIGH_INFLATION",
    "regime_NORMAL",
    "regime_EASING",
    "regime_TIGHTENING",
    "event_type_CPI",
    "event_type_FOMC",
    "event_type_FED_CUTS",
    "event_type_NFP",
)

PMPS_FEATURES: Tuple[str, ...] = (
    "PMPS_pre_Kalshi",
    "PMPS_pre_Polymarket",
)

MOMENT_FEATURES: Tuple[str, ...] = (
    "Delta_mean_pre",
    "Delta_variance_pre",
    "Delta_skew_pre",
)

LIQUIDITY_FEATURES: Tuple[str, ...] = (
    "PMPS_pre_weighted_Kalshi",
    "PMPS_pre_weighted_Polymarket",
    "delta_volume_pre_Kalshi",
    "delta_volume_pre_Polymarket",
    "delta_oi_pre_Kalshi",
)

# Nested configs built in feature_sets.py
CONFIG_IDS: Tuple[str, ...] = ("M0", "M1", "M2", "M3")
ESTIMATOR_NAMES: Tuple[str, ...] = ("ols", "elasticnet", "gbdt")

# Expanding-window HP search grids (train-only)
ELASTICNET_ALPHAS: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
ELASTICNET_L1_RATIOS: Tuple[float, ...] = (0.1, 0.5, 0.9)
N_EXPANDING_FOLDS: int = 3

LGBM_PARAM_GRID: Tuple[Dict, ...] = (
    {"n_estimators": 80, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 15},
    {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 31},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "num_leaves": 15},
)


@dataclass
class TrainConfig:
    """Runtime config for training."""

    ml_dir: Path = field(default_factory=lambda: DEFAULT_ML_DIR)
    cleaned_dir: Path = field(default_factory=lambda: DEFAULT_CLEANED_DIR)
    models_dir: Path = field(default_factory=lambda: DEFAULT_MODELS_DIR)
    target: str = DEFAULT_TARGET
    n_expanding_folds: int = N_EXPANDING_FOLDS
    random_state: int = 42

    def with_overrides(self, **kwargs) -> "TrainConfig":
        return replace(self, **kwargs)


DEFAULT_TRAIN_CONFIG = TrainConfig()
