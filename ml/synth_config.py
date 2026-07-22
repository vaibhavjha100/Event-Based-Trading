"""Configuration for synthetic prediction-market / equity dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Tuple

SCHEMA_VERSION = "1.0.0"

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw"

# Assets
EQUITY_ASSETS: Tuple[str, ...] = ("SPY", "QQQ", "XLF", "XLK", "XLU")
RETURN_WINDOWS: Tuple[str, ...] = ("5m", "30m", "60m", "EOD")
WINDOW_MINUTES: Dict[str, int] = {"5m": 5, "30m": 30, "60m": 60, "EOD": 390}

# Band probability sample offsets relative to release_time (minutes)
BAND_TIME_OFFSETS: Tuple[int, ...] = (-60, -30, -5, 30)

# Event-centered 1-min OHLCV window
OHLCV_PRE_MINUTES = 120
OHLCV_POST_MINUTES = 60


@dataclass
class SynthConfig:
    """Runtime configuration for the synthetic data pipeline."""

    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"
    seed: int = 42
    out_dir: Path = field(default_factory=lambda: DEFAULT_OUT_DIR)
    schema_version: str = SCHEMA_VERSION

    # Event calendar rates (approx per year)
    cpi_per_year: int = 12
    fomc_per_year: int = 8
    nfp_per_year: int = 12
    noise_events_per_year: float = 1.5

    # Latent factor distributions
    z_std: float = 1.0
    v_lognormal_mean: float = 0.0
    v_lognormal_sigma: float = 0.45
    l_lognormal_mean: float = 0.5
    l_lognormal_sigma: float = 0.55

    # Consensus / surprise
    surprise_beta: float = 0.55  # Surprise ≈ beta * Z + noise
    consensus_noise_std: float = 0.25
    actual_noise_std: float = 0.15

    # Platform view noise
    platform_eps_std: float = 0.35

    # Edge-case contract counts
    n_noise_contracts: int = 4
    n_malformed_contracts: int = 3
    n_overlap_unlinked_contracts: int = 3

    # OHLCV windows
    ohlcv_pre_minutes: int = OHLCV_PRE_MINUTES
    ohlcv_post_minutes: int = OHLCV_POST_MINUTES

    # Return model: base linear betas (SPY-like); scaled per asset below
    # R = b0 + b1*S + b2*P + b3*D + b4*VIX + b5*prior
    #   + b6*S^2 + b7*P^2 + b8*VIX^2 + b9*S*P
    #   + b10*P^3 + eps
    return_betas_spy: Tuple[float, ...] = (
        0.0002,   # b0
        -0.0045,  # b1 Surprise
        -0.0120,  # b2 PMPS_pre_K
        -0.0030,  # b3 disagreement
        -0.00015, # b4 VIX
        0.15,     # b5 prior_5d
        -0.0010,  # b6 S^2
        -0.0080,  # b7 P^2
        0.000002, # b8 VIX^2
        -0.0060,  # b9 S*P
        -0.0100,  # b10 P^3
    )

    # Multipliers vs SPY for other assets (on shock-related betas)
    asset_sensitivity: Dict[str, float] = field(
        default_factory=lambda: {
            "SPY": 1.0,
            "QQQ": 1.25,
            "XLF": 1.10,
            "XLK": 1.30,
            "XLU": 0.45,
        }
    )

    # Event-type extras for XLF (FOMC/rates) and XLK (cuts)
    xlf_fomc_extra: float = 1.4
    xlk_fed_cuts_extra: float = 1.35

    # Regime multipliers on Surprise and PMPS coefficients
    regime_shock_mult: Dict[str, float] = field(
        default_factory=lambda: {
            "HIGH_INFLATION": 1.35,
            "TIGHTENING": 1.25,
            "NORMAL": 1.0,
            "EASING": 0.85,
        }
    )

    # Asymmetry: amplify negative returns when shock adverse
    adverse_asymmetry: float = 1.35

    # Heteroskedastic noise scale
    return_noise_base: float = 0.0018

    def with_overrides(
        self,
        *,
        seed: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        out_dir: Path | str | None = None,
    ) -> "SynthConfig":
        """Return a copy with selected fields overridden."""
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        if out_dir is not None:
            kwargs["out_dir"] = Path(out_dir)
        return replace(self, **kwargs)


DEFAULT_CONFIG = SynthConfig()
