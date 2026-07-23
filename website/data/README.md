# Website data contract

The app runs with clearly labelled demo data until the required model, simulated
feature and backtest artifacts are present in this directory. Do not rename
fields without agreeing the change with the Website Lead.

## Signal handoff options

The instructor confirmed that a live Kalshi API feed is not required for the
final submission. The production demo may use simulated feature rows, provided
the simulation assumptions are disclosed and the model/backtest evidence remains
clearly separated from simulated user input.

### Option A: Data supplies precomputed signal rows

Provide `live_signals.csv` with the schema below.

### Option B: Website performs inference from the trained model

Provide all of the following files:

- `simulated_features.csv` preferred, or `latest_features.csv` for backward compatibility
- `model_metadata.json`
- `model_bundle.joblib` or `model_bundle.pkl`
- `cumulative_returns.csv`
- `model_metrics.json`
- `simulation_assumptions.md`

In this path, Data owns model training, preprocessing design, feature
definitions, simulated feature generation, backtest results and model
validation. Website owns artifact loading, schema validation, `predict()`
inference, display and error states.

`model_bundle.pkl` is treated as a trusted local team artifact only. It must not
be exposed as a public user-upload feature.

## live_signals.csv

Required columns:

- `date`
- `market_title`
- `category`
- `target_asset`
- `p_bad`
- `pmps_pre`
- `volume`
- `open_interest`
- `signal_direction`
- `model_forecast`
- `classification_confidence`
- `known_at`

## simulated_features.csv or latest_features.csv

Required metadata columns:

- `date`
- `market_title`
- `category`
- `target_asset`
- `p_bad`
- `pmps_pre`
- `volume`
- `open_interest`
- `classification_confidence`
- `known_at`

It must also include every model feature listed in
`model_metadata.json.feature_columns`, in exactly the same units and
preprocessing convention used during training.

`simulated_features.csv` should represent plausible macro scenario rows, not a
live market feed. Each row must still include a `known_at` timestamp for audit,
but Data confirmed that the current feature set includes `surprise`, which is
not available at the present release decision time. The Website must label these
rows as event-window scenario inputs.

## simulation_assumptions.md

Required content:

- simulation purpose and scope;
- which variables are simulated;
- plausible value ranges and how they were chosen;
- number of scenario rows;
- random seed if randomness is used;
- statement that simulated rows are for inference demonstration only;
- statement that empirical claims come from held-out backtest files, not from
  simulated signal rows.

## model_metadata.json

Required keys:

```json
{
  "model_version": "v1.0.0",
  "feature_columns": ["pmps_pre", "p_bad", "volume", "open_interest"],
  "prediction_kind": "continuous_return",
  "thresholds": {
    "tau": 0.001
  }
}
```

`prediction_kind` may be `return_forecast` or `continuous_return`. Thresholds
may be supplied as `long_threshold` / `short_threshold` or as a symmetric
`thresholds.tau`. The Website converts the predicted target return into Long,
Neutral or Short.

## cumulative_returns.csv

Required columns:

- `date`
- `strategy`
- `benchmark`
- `drawdown`
- `strategy_return`
- `benchmark_return`

The `strategy` and `benchmark` columns are wealth indices that start at 100.

The Website also accepts the Data Lead wide walk-forward format:

- `release_time`
- one cumulative-return column per model, such as `M0_ols`, `M2_ols`,
  `M3_gbdt`
- `selected`

In this format, the Website uses `model_metadata.json.model_id` as the displayed
strategy column and `M0_ols` as the benchmark when available.

## event_results.csv

Required for the final event-study and ablation tables:

- `event_id`
- `release_time`
- `event_type`
- `target_asset`
- `p_bad_t_minus_60m`
- `p_bad_t_minus_5m`
- `p_bad_t_plus_30m`
- `pmps_pre`
- `pmps_reaction`
- `actual`
- `consensus`
- `consensus_surprise`
- `return_5m`
- `return_30m`
- `return_60m`
- `return_eod`
- `vix_t_minus_1`
- `prior_5d_return`
- `volume_change`
- `open_interest_change`
- `regime`
- `model_version`

`pmps_pre` must use only timestamps before the release. `pmps_reaction` may use
post-release data but cannot be used to predict an overlapping return.

## model_comparison.csv

Required columns:

- `model_id`
- `feature_set`
- `split`
- `r2`
- `rmse`
- `directional_accuracy`
- `n_events`

At minimum, include M0 (consensus surprise plus controls) and M1 (M0 plus
`pmps_pre`).

## user_testing_results.csv

Required columns:

- `tester_id`
- `role`
- `task_id`
- `success`
- `time_seconds`
- `ease_1_to_5`
- `confidence_1_to_5`
- `issue`
- `suggestion`

At least three testers must complete the same structured tasks.

## model_metrics.json, or adapted model summary files

Required keys:

```json
{
  "model_version": "v1.0.0",
  "data_updated_at": "2024-12-31 16:00 America/New_York",
  "source": "Kalshi plus approved ETF price source",
  "train_period": "2020-01-01 to 2022-12-31",
  "validation_period": "2023-01-01 to 2023-06-30",
  "test_period": "2023-07-01 to 2024-12-31",
  "strategy_return": 0.126,
  "benchmark_return": 0.091,
  "sharpe": 1.18,
  "max_drawdown": -0.072,
  "hit_rate": 0.56,
  "number_of_trades": 84,
  "transaction_cost_bps": 10,
  "test_r2": 0.041,
  "test_directional_accuracy": 0.554
}
```

If the flat metric JSON does not contain these keys, the Website derives display
metrics from `model_comparison.csv`, `model_metadata.json` and
`data_quality_summary.csv`. This is the current Data Lead handoff path.

Current caveats to disclose:

- `simulated_features.csv` and `prediction_fixture.csv` are demo-only scenario
  rows.
- The current website model is `M2_ols`, confirmed by the Data Lead because it
  has the second-highest selection score and avoids native deployment risk.
- The current target is `ret_SPY_60m`.
- The current scope is SPY only.
- The current backtest uses no transaction costs or slippage.
- `surprise` is not available at the present release decision time, so the
  current model is event-window evidence rather than a fully pre-release trading
  signal.
- The date range is no longer constrained by the earlier 2020-2024 range because
  the final website demo uses simulated scenario data plus the exported
  walk-forward sample.

## Publication rule

Demo values must never be presented as empirical findings. The Website Lead may
remove the generic demo warning only after the Data Lead confirms that the
model, simulated feature panel and backtest files come from the frozen handoff
and that the Finance Lead has reviewed the interpretation. The Website should
still label the signal input as simulated wherever applicable.
