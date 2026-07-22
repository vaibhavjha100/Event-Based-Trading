"""Column schemas for synthetic dataset outputs."""

from __future__ import annotations

from typing import Dict, List

EVENTS_COLUMNS: List[str] = [
    "event_id",
    "event_type",
    "macro_variable",
    "release_time",
    "regime",
    "day_of_week",
]

EVENTS_LATENT_COLUMNS: List[str] = [
    "event_id",
    "Z_e",
    "V_e",
    "L_e",
]

CONSENSUS_COLUMNS: List[str] = [
    "event_id",
    "macro_variable",
    "consensus_value",
    "consensus_time",
    "surprise_value",
    "actual_value",
]

BAND_PROB_COLUMNS: List[str] = [
    "event_id",
    "platform",
    "band_id",
    "band_value",
    "band_label",
    "is_adverse",
    "timestamp",
    "minutes_to_release",
    "probability",
]

KALSHI_CONTRACT_COLUMNS: List[str] = [
    "ticker",
    "event_ticker",
    "market_type",
    "yes_sub_title",
    "title",
    "status",
    "last_price",
    "yes_bid",
    "yes_ask",
    "no_bid",
    "no_ask",
    "volume",
    "volume_24h",
    "open_interest",
    "result",
    "created_time",
    "open_time",
    "close_time",
    "event_id",
    "band_id",
    "edge_case_type",
]

POLYMARKET_CONTRACT_COLUMNS: List[str] = [
    "id",
    "condition_id",
    "question",
    "slug",
    "outcomes",
    "outcome_prices",
    "clob_token_ids",
    "volume",
    "liquidity",
    "active",
    "closed",
    "end_date",
    "created_at",
    "market_maker_address",
    "event_id",
    "band_id",
    "edge_case_type",
]

CONTRACT_EVENT_MAP_COLUMNS: List[str] = [
    "platform",
    "contract_id",
    "event_id",
    "macro_variable",
    "band_id",
    "is_core_macro_contract",
    "edge_case_type",
]

KALSHI_OHLCV_COLUMNS: List[str] = [
    "timestamp",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
]

POLYMARKET_OHLCV_COLUMNS: List[str] = [
    "timestamp",
    "condition_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

EVENT_SIGNALS_COLUMNS: List[str] = [
    "event_id",
    "PMPS_pre_K",
    "PMPS_pre_P",
    "PMPS_pre_weighted_K",
    "PMPS_pre_weighted_P",
    "PMPS_reaction_K",
    "PMPS_reaction_P",
    "p_bad_K_t0_60",
    "p_bad_K_t0_5",
    "p_bad_P_t0_60",
    "p_bad_P_t0_5",
    "Delta_mean_pre",
    "Delta_variance_pre",
    "Delta_skew_pre",
    "disagreement_pre",
    "delta_volume_pre_K",
    "delta_volume_pre_P",
    "delta_oi_pre_K",
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
    "event_type_NOISE",
    "surprise_value",
]

DAILY_EQUITY_COLUMNS: List[str] = [
    "date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

EQUITY_1MIN_COLUMNS: List[str] = [
    "timestamp",
    "event_id",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

EVENT_RETURNS_COLUMNS: List[str] = [
    "event_id",
    "asset",
    "window",
    "return_value",
]

VIX_DAILY_COLUMNS: List[str] = [
    "date",
    "vix_close",
]

SCHEMA_BY_FILE: Dict[str, List[str]] = {
    "events.csv": EVENTS_COLUMNS,
    "events_latent.csv": EVENTS_LATENT_COLUMNS,
    "consensus_forecasts.csv": CONSENSUS_COLUMNS,
    "event_band_probabilities.csv": BAND_PROB_COLUMNS,
    "kalshi_contracts.csv": KALSHI_CONTRACT_COLUMNS,
    "polymarket_contracts.csv": POLYMARKET_CONTRACT_COLUMNS,
    "contract_event_map.csv": CONTRACT_EVENT_MAP_COLUMNS,
    "kalshi_1min_ohlcv.csv": KALSHI_OHLCV_COLUMNS,
    "polymarket_1min_ohlcv.csv": POLYMARKET_OHLCV_COLUMNS,
    "event_signals.csv": EVENT_SIGNALS_COLUMNS,
    "daily_equity_prices.csv": DAILY_EQUITY_COLUMNS,
    "equity_1min_ohlcv.csv": EQUITY_1MIN_COLUMNS,
    "event_returns.csv": EVENT_RETURNS_COLUMNS,
    "vix_daily.csv": VIX_DAILY_COLUMNS,
}
