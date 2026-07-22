# Preprocessing pipeline

## Purpose

Clean, classify, align, and reshape raw prediction-market / macro-event data into
signal-ready tables for walk-forward ML and event studies.

Outputs are written under `data/cleaned/`.

## How to run

```bash
pip install -r requirements.txt
python -m ml.run_preprocessing
python -m ml.run_preprocessing --skip-gemini
python -m ml.run_preprocessing --gemini-failed-only
python -m ml.run_preprocessing --force-gemini
```

Requires `GENAI_API_KEY` in `.env` for Gemini calls. `--skip-gemini` uses
deterministic rule-based classification and band parsing (fully offline).

If the Gemini API returns quota errors (HTTP 429), the pipeline disables further
Gemini calls for that run and falls back to rules automatically.

## Stages

1. Load / standardize schemas and UTC timestamps
2. Gemini (or rules): family-level contract type + contract-level band extraction
3. Contract–event map: use `data/raw/contract_event_map.csv` if present, else rebuild
4. Universe selection (primary vs research) + exclusions audit
5. Band semantics (signed distance vs latest pre-release consensus, adverse flags)
6. Split PM OHLCV into pre-release (`<= t0-5`) and reaction windows
7. Recompute leakage-safe signals (PMPS_pre, moments, liquidity, disagreement)
8. Recompute event returns from equity 1-min bars (source of truth)
9. Build `event_level_ml_base.csv` + validation report

## Gemini usage

- **Tier 1 (family):** coarse type `CPI` / `FOMC` / `FED_CUTS` / `OTHER_MACRO` / `NON_MACRO` / `AMBIGUOUS`
- **Tier 2 (contract):** band threshold / directionality for signed distance and `p_bad`
- Structured JSON schema via `google-genai` when enabled
- Cache: `data/cleaned/cache/gemini_classifications.jsonl`

## Consensus rule

Latest pre-release consensus = most recent `consensus_time` with `consensus_time < t0`.

Adverse outcomes: `signed_distance = band_value - consensus > 0` (hotter / hawkish).
Never treat YES as adverse by default.

## Leakage rules

| Feature set | Allowed timestamps |
|-------------|-------------------|
| Predictive (`PMPS_pre_*`, moments_pre, liquidity_pre, disagreement_pre) | only `<= t0-5` |
| Reaction (`reaction_*`) | t0-30 to t0+30; event-study only |

Predictive columns are listed in `preprocessing_report.json` under
`predictive_feature_columns`. Reaction columns must not be used for overlapping
return forecasting.

## Universes

- **Primary:** CPI, FOMC, FED_CUTS; usable; not malformed; retained; no edge cases
- **Research:** primary + OTHER_MACRO (e.g. NFP)
- **Excluded:** NON_MACRO, noise, malformed, timestamp-overlap-unlinked, low confidence

## Main outputs

| File | Role |
|------|------|
| `events_cleaned.csv` | Clean event calendar |
| `consensus_cleaned.csv` | Clean consensus forecasts |
| `kalshi_contracts_cleaned.csv` / `polymarket_contracts_cleaned.csv` | Cleaned metadata |
| `contracts_normalized.csv` | Platform-unified contracts |
| `contract_classifications.csv` | Gemini/rules extraction + review status |
| `contract_event_map_cleaned.csv` | Contract ↔ event map |
| `band_semantics_cleaned.csv` | Band, signed distance, adverse, inclusion |
| `universe_primary_contracts.csv` / `universe_research_contracts.csv` | Universes |
| `exclusions_audit.csv` | Drop reasons |
| `pm_ohlcv_prerelease.csv` / `pm_ohlcv_reaction.csv` | Leakage-separated PM bars |
| `band_probs_cleaned.csv` | Cleaned band probabilities |
| `vix_cleaned.csv` / `daily_equity_cleaned.csv` | Controls sources |
| `event_returns_cleaned.csv` | Returns from equity 1-min |
| `event_level_ml_base.csv` | Event-level ML-ready table |
| `preprocessing_report.json` | QA summary |

## Assumptions / exclusions

- Does not use `events_latent.csv` or raw `event_signals.csv` as features
- Raw `event_returns.csv` is a cross-check only; cleaned returns come from equity 1-min
- Minute PM/equity bars remain event-windowed (see synthetic data README)
- Primary analysis excludes noise, malformed bands, and timestamp-overlap-unlinked contracts
