# ML dataset preparation

## Purpose

Build leakage-safe feature/target matrices from the cleaned event-level table and
apply a **chronological** 75:25 train/test split (no shuffle).

Outputs go to `data/ml/`.

## How to run

```bash
python -m ml.prepare_ml_data
python -m ml.prepare_ml_data --train-ratio 0.75 --cleaned-dir data/cleaned --out-dir data/ml
```

Requires `data/cleaned/event_level_ml_base.csv` from preprocessing.

## Split rule

1. Drop `event_type == "NOISE"`.
2. Sort by `release_time` ascending.
3. Keep rows with non-null primary targets (`ret_SPY_60m`, `ret_QQQ_60m`) and
   non-null **core** features.
4. Earliest 75% of remaining rows → train; latest 25% → test.

## Feature policy

| Group | Rule |
|-------|------|
| **Core** | Must exist; rows require non-null values |
| **Optional** | Included if column present; null cells allowed |

Core: `PMPS_pre_Kalshi`, Kalshi `p_bad` at t0-60/t0-5, `consensus_value`,
`VIX_Tminus1`, `prior_5d_SPY_return`.

Optional: Polymarket PMPS/p_bad, weighted PMPS, distributional moments,
disagreement, pre-release liquidity deltas, regime and event-type indicators
(CPI/FOMC/FED_CUTS/NFP).

### Excluded (leakage / non-features)

- `reaction_*` (contemporaneous event-study only)
- `ret_*` (targets)
- `actual_value`, `surprise` (use realized print; not known by t0-5)
- `event_type_NOISE` and raw IDs / timestamps / categoricals

## Outputs

| File | Role |
|------|------|
| `X_train.csv` / `X_test.csv` | Feature matrices |
| `y_train.csv` / `y_test.csv` | Target matrices |
| `feature_columns.csv` | Feature name + `group` (core/optional) |
| `target_columns.csv` | Target names |
| `train_event_ids.csv` / `test_event_ids.csv` | Event IDs and release times |
| `ml_dataset_full.csv` | Filtered chronologically ordered table |
| `split_metadata.json` | Counts, boundary, column lists |

## Not in scope

Model training, scaling/encoding, and walk-forward CV folds (future work).
