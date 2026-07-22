# Walk-forward backtest and model selection

Score all trained `M0`–`M3` × `{ols, elasticnet, gbdt}` artifacts on the chronological test set, select one model with an explicit multi-metric rule, and export research plus website handoff files under `output/`.

## Run

```bash
python -m ml.run_backtest
```

Optional: `--output-dir path/to/dir` (default: `output/`).

## Walk-forward mode

`walkforward_mode = "expanding_refit_fixed_hparams"`

For each model and each test event `i` in chronological order:

1. Design matrix = original train rows + test rows `0..i-1` only (strictly past).
2. Clone a fresh pipeline of the same type as the saved artifact, using stored HPs (`best_alpha` / `best_l1_ratio` / `best_params`; OLS has none).
3. Fit median imputer + estimator on that expanding history.
4. Predict only event `i`.

Hyperparameters are **not** re-tuned on test data. Target: `ret_SPY_60m`. Feature lists come from each artifact’s stored `feature_names` (no `reaction_*` / other leakage columns). `surprise` is joined from `data/cleaned/event_level_ml_base.csv` by `event_id` for both train and test scoring frames.

## Signals and PnL

- Fit once on full original train; in-sample `ŷ_train`.
- `τ = max(1e-6, 0.5 * std(ŷ_train))` (train predictions only).
- Long if `ŷ >= +τ`; Short if `ŷ <= -τ`; else Neutral.
- Event PnL: Long → `+R`, Short → `−R`, Neutral → `0` (`R` = realized `ret_SPY_60m`).
- **No transaction costs.** Entry/exit aligned to the target window (t0−5 → t0+60).

## Model selection rule

Among models with `n_trades >= 5`:

`score = 0.40 * directional_accuracy + 0.35 * sharpe_like_clipped + 0.25 * r2_clipped`

where:

- `directional_accuracy` ∈ [0, 1] on test events with non-null pred/actual
- `sharpe_like_clipped = clip( mean(pnl) / (std(pnl)+eps) / 3 , 0, 1)` on trade PnLs (fallback: all-event PnL)
- `r2_clipped = clip(R², 0, 1)` (negative R² → 0 for selection only; raw R² still reported)

If no model has ≥5 trades, drop the trade filter and use the same score. Tie-break: higher cumulative return, then lower RMSE. The full reason is written to `model_metadata.json.selection_reason`.

## Outputs (`output/`)

### Research (historical walk-forward test period)

| File | Description |
|------|-------------|
| `event_results.csv` | Event × model predictions, signals, PnL |
| `cumulative_returns.csv` | Cumulative PnL by `model_id` + `selected` |
| `model_metrics.json` | Per-model metrics |
| `model_comparison.csv` | Leaderboard with selection score |
| `trade_log.csv` | Non-neutral decisions |
| `data_quality_summary.csv` | Counts, skips, mode, exclusions |

### Website handoff

| File | Description |
|------|-------------|
| `model_bundle.pkl` | Selected artifact + `SklearnReturnModel.predict(X)`; `preprocessing_included=true` |
| `model_metadata.json` | Version, features, thresholds, periods, packages, selection reason, walkforward mode |
| `simulated_features.csv` | **Demo-only** rows (not historical): website columns + all `feature_columns` |
| `simulation_assumptions.md` | Seed, ranges, disclaimer |
| `prediction_fixture.csv` | Exact features + `expected_prediction` / `expected_signal` for integration checks |

Historical walk-forward rows are labeled in metadata / quality summary as `historical_walkforward_test`. Simulated rows are labeled `demo_only_not_historical`.

## Layout

```
ml/
  backtest_config.py
  run_backtest.py
  backtesting/
    load_artifacts.py
    prepare_test_data.py
    walkforward.py
    signal_policy.py
    metrics.py
    select_best_model.py
    website_bundle.py
    io_utils.py
  README_backtesting.md
```

## Validation (run automatically)

- Expanding refit never includes current/future test rows
- Leakage column audit on selected feature list
- Bundle load + fixture predictions match exactly
- `simulated_features` schema ⊇ metadata `feature_columns`
- All 11 output files present under `output/`
