# Simulation assumptions

## Purpose
`simulated_features.csv` and `prediction_fixture.csv` are **demo-only** rows for website
integration testing. They are **not** historical walk-forward backtest events.

## Generation method
- Fixed seed: `7`
- Rows: `10` simulated feature rows; fixture uses first `8`
- Website columns: date, market_title, category, target_asset, p_bad, pmps_pre, volume,
  open_interest, classification_confidence, known_at
- Model feature columns: sampled uniformly from approx. 5th–95th percentile ranges of the
  training design matrix (one-hot regime/event flags drawn as 0/1; surprise ~ N(0, 0.5))
- `expected_prediction` / `expected_signal` come from `model_bundle.predict` and train-only τ

## Ranges
Feature ranges are derived from the selected model's training feature matrix percentiles.
Website liquidity fields (volume, open_interest) are synthetic integers for UI display only.
The selected `M2_ols` feature set includes `surprise`, which is not available at the
present release decision time. These rows support event-window inference demonstration,
not a fully pre-release trading signal.

## Trading assumptions (backtest, not demo)
- Entry/exit aligned to target definition: t0−5 → t0+60 for `ret_SPY_60m`
- No transaction costs, no slippage
- Long / Short / Neutral from train-only threshold τ
- Walk-forward mode: `expanding_refit_fixed_hparams`

## Disclaimer
Do not treat simulated rows as live market data or as out-of-sample research results.
Historical research outputs are labeled separately (`event_results.csv`, etc.).
