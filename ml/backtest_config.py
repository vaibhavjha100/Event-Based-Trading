"""Configuration for walk-forward backtest and website handoff."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DIR = PROJECT_ROOT / "data" / "ml"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

EVENT_LEVEL_PATH = CLEANED_DIR / "event_level_ml_base.csv"
SPLIT_METADATA_PATH = ML_DIR / "split_metadata.json"

PRIMARY_TARGET = "ret_SPY_60m"
WALKFORWARD_MODE = "expanding_refit_fixed_hparams"

CONFIG_IDS = ["M0", "M1", "M2", "M3"]
ESTIMATORS = ["ols", "elasticnet", "gbdt"]

# Signal thresholds from train-only predictions
TAU_STD_FRAC = 0.5
TAU_FLOOR = 1e-6

# Selection rule weights
W_DIR_ACC = 0.40
W_SHARPE = 0.35
W_R2 = 0.25
MIN_TRADES = 5
SHARPE_CLIP_SCALE = 3.0
EPS = 1e-12

# Website demo simulation
SIM_SEED = 7
N_SIMULATED_ROWS = 10
N_FIXTURE_ROWS = 8

LEAKAGE_PREFIXES = ("reaction_", "ret_")
LEAKAGE_EXACT = frozenset({"actual_value"})

PACKAGE_VERSION_KEYS = ("numpy", "pandas", "scikit-learn", "lightgbm")
