# Model training (train-only)

## Purpose

Fit nested methodology configurations **M0–M3** on the ML training partition
only. No test-set access. Artifacts go to `models/`.

## How to run

```bash
python -m ml.train_models
python -m ml.train_models --target ret_SPY_60m --models-dir models
```

Requires `data/ml/X_train.csv`, `y_train.csv`, `train_event_ids.csv`, and
`data/cleaned/event_level_ml_base.csv` (for joining `surprise` onto train ids).

## Configurations

| Config | Features |
|--------|----------|
| M0 | `surprise` + controls |
| M1 | M0 + PMPS_pre (Kalshi/Polymarket) |
| M2 | M1 + distributional moments |
| M3 | M2 + liquidity-weighted / volume / OI features |

Default target: `ret_SPY_60m`.

## Estimators

For each config: OLS, Elastic Net, LightGBM (GBDT; HistGradientBoosting fallback).

Each artifact is a pickle dict containing:
`pipeline` (imputer + model), `feature_names`, `config_id`, `estimator_name`,
`target`, `n_rows_used`.

## Train-only rules

- Never load `X_test` / `y_test`.
- Join `surprise` only for `train_event_ids`.
- Verify `release_time` is non-decreasing after the join.
- Per config: keep rows with non-null target and ≥1 non-null feature; median-impute the rest.
- Hyperparameter search uses expanding-window splits on training rows only.

## Outputs

- `models/M{k}_{ols|elasticnet|gbdt}.pkl`
- `models/training_summary.json` (exact per-model feature lists)
- `models/coefficients_summary.csv` (linear models)

## Not in scope

Test-set evaluation and holdout metrics (next stage).
