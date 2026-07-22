"""Configuration for ML dataset preparation and chronological split."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
DEFAULT_ML_DIR = PROJECT_ROOT / "data" / "ml"

EVENT_LEVEL_FILENAME = "event_level_ml_base.csv"

# Core features: must exist; rows require non-null values
CORE_FEATURE_COLUMNS: Tuple[str, ...] = (
    "PMPS_pre_Kalshi",
    "p_bad_Kalshi_t0_60",
    "p_bad_Kalshi_t0_5",
    "consensus_value",
    "VIX_Tminus1",
    "prior_5d_SPY_return",
)

# Optional: include if column present; null cells allowed
OPTIONAL_FEATURE_COLUMNS: Tuple[str, ...] = (
    "PMPS_pre_Polymarket",
    "PMPS_pre_weighted_Kalshi",
    "PMPS_pre_weighted_Polymarket",
    "p_bad_Polymarket_t0_60",
    "p_bad_Polymarket_t0_5",
    "Delta_mean_pre",
    "Delta_variance_pre",
    "Delta_skew_pre",
    "disagreement_pre",
    "delta_volume_pre_Kalshi",
    "delta_volume_pre_Polymarket",
    "delta_oi_pre_Kalshi",
    "regime_HIGH_INFLATION",
    "regime_NORMAL",
    "regime_EASING",
    "regime_TIGHTENING",
    "event_type_CPI",
    "event_type_FOMC",
    "event_type_FED_CUTS",
    "event_type_NFP",
)

PRIMARY_TARGET_COLUMNS: Tuple[str, ...] = (
    "ret_SPY_60m",
    "ret_QQQ_60m",
)

# Easy expansion; not exported by default unless requested
ADDITIONAL_TARGET_COLUMNS: Tuple[str, ...] = (
    "ret_SPY_5m",
    "ret_SPY_30m",
    "ret_SPY_EOD",
    "ret_QQQ_5m",
    "ret_QQQ_30m",
    "ret_QQQ_EOD",
    "ret_XLF_60m",
    "ret_XLK_60m",
    "ret_XLU_60m",
)

EXPLICIT_LEAKAGE_COLUMNS: Tuple[str, ...] = (
    "actual_value",
    "surprise",
    "event_type_NOISE",
)

EXCLUDE_EVENT_TYPES: Tuple[str, ...] = ("NOISE",)


@dataclass
class MLDataConfig:
    """Runtime config for prepare_ml_data."""

    cleaned_dir: Path = field(default_factory=lambda: DEFAULT_CLEANED_DIR)
    out_dir: Path = field(default_factory=lambda: DEFAULT_ML_DIR)
    train_ratio: float = 0.75
    event_level_filename: str = EVENT_LEVEL_FILENAME
    core_features: Tuple[str, ...] = CORE_FEATURE_COLUMNS
    optional_features: Tuple[str, ...] = OPTIONAL_FEATURE_COLUMNS
    primary_targets: Tuple[str, ...] = PRIMARY_TARGET_COLUMNS
    additional_targets: Tuple[str, ...] = ADDITIONAL_TARGET_COLUMNS
    include_additional_targets: bool = False
    exclude_event_types: Tuple[str, ...] = EXCLUDE_EVENT_TYPES

    def with_overrides(self, **kwargs) -> "MLDataConfig":
        return replace(self, **kwargs)

    @property
    def event_level_path(self) -> Path:
        return self.cleaned_dir / self.event_level_filename


DEFAULT_ML_DATA_CONFIG = MLDataConfig()
