# Synthetic prediction-market / equity dataset

## Purpose

This package generates a fully synthetic but internally consistent dataset for
event-based trading research: macro event calendars, Kalshi/Polymarket-style
contracts, band probabilities, derived signals, equities, and VIX.

All outputs are written under `data/raw/`.

## How to run

```bash
pip install -r requirements.txt
python -m ml.generate_synthetic_data
python -m ml.generate_synthetic_data --seed 7 --start 2021-01-01 --end 2025-12-31
```

Configuration lives in `ml/synth_config.py` (`SCHEMA_VERSION`, horizon, betas, edge-case counts).

## Output files

| File | Role |
|------|------|
| `events.csv` | Master event calendar (CPI, FOMC, FED_CUTS, NFP, NOISE) |
| `events_latent.csv` | Hidden factors `Z_e`, `V_e`, `L_e` (do not use as model features) |
| `consensus_forecasts.csv` | Consensus, Actual, Surprise |
| `event_band_probabilities.csv` | Discrete band probs by platform / time |
| `kalshi_contracts.csv` | Kalshi-like contract metadata |
| `polymarket_contracts.csv` | Polymarket-like contract metadata |
| `contract_event_map.csv` | Explicit join keys: platform contract ↔ event / band / edge type |
| `kalshi_1min_ohlcv.csv` | Event-window 1-min YES OHLCV |
| `polymarket_1min_ohlcv.csv` | Event-window 1-min OHLCV |
| `daily_equity_prices.csv` | Full-horizon daily OHLCV for SPY/QQQ/XLF/XLK/XLU |
| `vix_daily.csv` | Full-horizon daily VIX |
| `event_signals.csv` | PMPS, moments, disagreement, lagged controls |
| `event_returns.csv` | Per-event / asset / window log-returns |
| `equity_1min_ohlcv.csv` | Event-window equity 1-min bars (`event_id` tagged) consistent with returns |
| `generation_manifest.json` | Row counts, schema version, validation messages |

## Generative assumptions

1. Latent adverse shock `Z_e ~ N(0,1)` drives consensus surprise, prediction-market
   adverse-band probability, and (via signals) equity returns.
2. Platform views add independent noise so Kalshi and Polymarket disagree mildly.
3. Returns use linear + quadratic + cubic terms in Surprise, PMPS_pre, disagreement,
   VIX, and prior 5-day SPY return, with regime / event-type interactions and
   heteroskedastic noise scaled by `V_e`.
4. **One-way flow:** `event_signals` is computed before `event_returns` and
   `equity_1min_ohlcv`. Signals never depend on event-window equity returns.

## OHLCV limitation (important)

Minute bars for prediction markets and equities are **intentionally generated
only for event-centered windows** (default: release −120m to release +60m).

Contract metadata still uses realistic multi-day `open_time` / `close_time`
(or Polymarket `created_at` / `end_date`). Full-horizon history for lagged
controls comes from `daily_equity_prices.csv` and `vix_daily.csv`, not from
minute bars.

## Edge-case contracts

Labeled in `contract_event_map.csv` via `edge_case_type`:

- `noise` — unrelated markets; `event_id` null; independent of `Z_e`
- `malformed_bands` — overlapping / non-monotonic band structure
- `timestamp_overlap_unlinked` — overlaps a macro release time but is not driven by that shock

Filter core training contracts with `is_core_macro_contract == True`.

## Known limitations

- Intraday microstructure is simplified (piecewise-linear anchors + noise).
- EOD return window is set to the realized path return at the end of the
  post-event minute window (coincides with 60m when that window is 60 minutes).
- No overnight gaps, exchange calendars, or corporate actions.
- Latent factors are available for diagnostics only; treat them as leaked if used in models.
