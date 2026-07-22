"""Configuration for the preprocessing / signal-ready pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"

PREPROCESS_SCHEMA_VERSION = "1.0.0"

PRIMARY_CONTRACT_TYPES = ("CPI", "FOMC", "FED_CUTS")
RESEARCH_EXTRA_TYPES = ("OTHER_MACRO",)

PREDICTIVE_FEATURE_COLUMNS: Tuple[str, ...] = (
    "PMPS_pre_Kalshi",
    "PMPS_pre_Polymarket",
    "PMPS_pre_weighted_Kalshi",
    "PMPS_pre_weighted_Polymarket",
    "p_bad_Kalshi_t0_60",
    "p_bad_Kalshi_t0_5",
    "p_bad_Polymarket_t0_60",
    "p_bad_Polymarket_t0_5",
    "Delta_mean_pre",
    "Delta_variance_pre",
    "Delta_skew_pre",
    "disagreement_pre",
    "surprise",
    "consensus_value",
    "VIX_Tminus1",
    "prior_5d_SPY_return",
)

REACTION_FEATURE_COLUMNS: Tuple[str, ...] = (
    "reaction_PMPS_Kalshi",
    "reaction_PMPS_Polymarket",
    "reaction_p_bad_Kalshi_t0_m30",
    "reaction_p_bad_Kalshi_t0_p30",
    "reaction_p_bad_Polymarket_t0_m30",
    "reaction_p_bad_Polymarket_t0_p30",
)

WINDOW_MINUTES: Dict[str, int] = {"5m": 5, "30m": 30, "60m": 60, "EOD": 60}


@dataclass
class PreprocessConfig:
    """Runtime config for preprocessing."""

    raw_dir: Path = field(default_factory=lambda: DEFAULT_RAW_DIR)
    cleaned_dir: Path = field(default_factory=lambda: DEFAULT_CLEANED_DIR)
    schema_version: str = PREPROCESS_SCHEMA_VERSION

    gemini_model: str = "gemini-2.0-flash"
    gemini_enabled: bool = True
    gemini_force: bool = False
    gemini_failed_only: bool = False
    confidence_min_primary: float = 0.45

    ohlcv_pre_minutes: int = 120
    ohlcv_post_minutes: int = 60

    primary_types: Tuple[str, ...] = PRIMARY_CONTRACT_TYPES
    research_extra_types: Tuple[str, ...] = RESEARCH_EXTRA_TYPES

    def cache_path(self) -> Path:
        return self.cleaned_dir / "cache" / "gemini_classifications.jsonl"

    def with_overrides(self, **kwargs) -> "PreprocessConfig":
        return replace(self, **kwargs)


DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()
