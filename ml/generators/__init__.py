"""Synthetic data generators for event-based trading research."""

from ml.generators.bands import generate_band_probabilities
from ml.generators.consensus import generate_consensus
from ml.generators.contract_map import build_contract_event_map
from ml.generators.equities import (
    generate_daily_equity_prices,
    generate_equity_1min_ohlcv,
    generate_event_returns,
)
from ml.generators.events import generate_events_and_latents
from ml.generators.kalshi import generate_kalshi_contracts, generate_kalshi_ohlcv
from ml.generators.polymarket import (
    generate_polymarket_contracts,
    generate_polymarket_ohlcv,
)
from ml.generators.signals import generate_event_signals
from ml.generators.validate import run_validations, write_manifest
from ml.generators.vix import generate_vix_daily

__all__ = [
    "generate_events_and_latents",
    "generate_consensus",
    "generate_band_probabilities",
    "generate_kalshi_contracts",
    "generate_kalshi_ohlcv",
    "generate_polymarket_contracts",
    "generate_polymarket_ohlcv",
    "build_contract_event_map",
    "generate_daily_equity_prices",
    "generate_vix_daily",
    "generate_event_signals",
    "generate_event_returns",
    "generate_equity_1min_ohlcv",
    "run_validations",
    "write_manifest",
]
