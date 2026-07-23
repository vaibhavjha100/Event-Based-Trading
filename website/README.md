# Team Prosper - Prediction Market Macro Signal Lab

Team Prosper is an AI/ML finance research application that tests whether
prediction-market probability shifts around scheduled macro releases improve
short-window equity-index forecasts beyond a standard
economist-consensus-surprise baseline. The current Data Lead handoff targets
`ret_SPY_60m` and includes CPI, FOMC, FED_CUTS and NFP scenario categories.

The deployed application is the inference, presentation and audit layer for a
chronological, walk-forward research pipeline. The final classroom demo may use
simulated model-ready scenario rows rather than a live Kalshi API feed. It does
not train the model, recompute the backtest, provide financial advice or execute
trades.

## Features

- Held-out performance dashboard with benchmark, drawdown and the exported transaction-cost assumption
- Auditable website-side model inference with simulated event-window scenario inputs
- Baseline-to-extension model comparison and ablation plan
- Structured market-classification agent with human-review guardrails
- Explicit disclosure that the current model includes macro surprise and should be interpreted as event-window evidence
- Data-quality, ethics, limitations and user-testing evidence
- Visible model version, data timestamp and demo/validated-output status
- Graceful fallback when required model, feature or backtest artifacts are missing or malformed

## Research design

The current Data Lead handoff is an event-window model. It uses macro surprise
and prediction-market probability features to forecast `ret_SPY_60m`.
`surprise` is not available at the present release decision time, so the current
model should not be presented as a fully pre-release trading signal.

The main prediction-market feature is the pre-release Prediction Market
Probability Shift:

```text
PMPS_pre = p_bad(t0 - 5 minutes) - p_bad(t0 - 60 minutes)
```

The primary target is the 60-minute event-window log return. A separate
`PMPS_reaction` measure may be used for contemporaneous event-study analysis and
should not be used to support a pre-release trading claim.

The core model ladder is:

1. M0: consensus surprise and controls
2. M1: M0 plus `PMPS_pre`
3. M2: M1 plus distribution moments; current selected website model is `M2_ols`
4. M3: liquidity-weighted robustness extension

Data confirmed that model selection uses:

```text
selection_score = 0.40 * directional_accuracy
                + 0.35 * sharpe_like_clipped
                + 0.25 * r2_clipped
```

Models are re-estimated using an expanding window. The held-out period is never
used for feature, threshold or model selection.

## Installation

Requirements:

- Python 3.10 or newer
- pip

```bash
git clone <TEAM_REPOSITORY_URL>
cd Event-Based-Trading
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r website/requirements.txt
```

## Environment setup

Copy `.env.example` to `.env` and populate only the keys required by the final
data pipeline. Never commit `.env` or credentials.

```bash
cp .env.example .env
```

The presentation layer reads validated local model, CSV and JSON artifacts and
does not require an API key for the final simulated-data demo. Future live
refreshes may use the environment variables documented in `.env.example`.

## Run

```bash
streamlit run website/app.py
```

Open `http://localhost:8501`.

## Worked example

1. Open **Scenario Signals**.
2. Filter to `CPI` or `FOMC` and `SPY`.
3. Select a market and confirm the adverse probability, direction and scenario timestamp.
4. Open **Methodology** to see why the current model is labelled event-window.
5. Open **Backtest** to inspect the selected `M2_ols` model and its baseline.
6. Open **Data & Limitations** to review data QA, ethical controls and the claim
   boundary.

Screenshots will be added after the final validated data and public deployment
are frozen.

## Data artifacts

See `data/README.md` for schemas. For displayed signals, the app can either
read a precomputed `live_signals.csv` or run inference from:

- `data/simulated_features.csv` preferred, or `data/latest_features.csv`
- `data/model_metadata.json`
- `data/model_bundle.joblib` or `data/model_bundle.pkl`
- `data/simulation_assumptions.md`

Backtest display still requires:

- `data/cumulative_returns.csv`
- `data/model_metrics.json`

The app displays a prominent demo warning if any required file or field is
missing. Demo values must never be reported as empirical findings.

## Architecture

```text
Kalshi trades + contract rules
             |
             v
Classification agent -> confidence / human review
             |
             v
Deterministic signal and timestamp pipeline
             |
             v
Macro consensus + SPY event-window matching
             |
             v
Walk-forward OLS / Elastic Net / robustness model
             |
             v
Model bundle + versioned CSV/JSON outputs -> Streamlit application
```

## Performance optimisation

Deterministic file reads and transforms are cached with `st.cache_data`. The app
loads bounded feature and summary artifacts rather than raw trade data, keeping
interactive filter changes fast and preventing training or backtest computation
in the UI layer.

## Security and responsible use

- No credentials or API keys are stored in source code.
- `.env` is excluded from Git.
- The application does not collect personal financial information.
- Market classification uses confidence thresholds and human review.
- The language agent does not calculate returns or fit models.
- Website inference loads only trusted local team artifacts; model uploads are
  not exposed as a public user feature.
- The application is a research prototype, not financial advice.

## Current handoff caveats

- Scenario rows are simulated for website-side inference demonstration; empirical claims come from historical walk-forward backtest artifacts.
- Data confirmed SPY-only scope for the current output.
- Data confirmed the exported backtest assumes nil transaction costs and no slippage.
- Data confirmed the website should use `M2_ols` because it has the second-highest selection score and avoids native deployment risk.
- Data confirmed `surprise` is not available at the present release decision time, so the current model is event-window evidence rather than a fully pre-release trading model.
- Public deployment URL, final screenshots and user-testing summary are not yet added.

## Planned enhancements

- Distributional mean, variance and skew signals for multi-bin CPI markets
- Liquidity-weighted PMPS
- Regime and sector-response views
- Validated user-testing summary
- Automated but versioned data refresh with stale-data alerts

## Academic and AI disclosure

External data, libraries, papers and APIs must be cited in the written report.
AI coding assistance must be disclosed, including the tasks for which it was
used. AI-generated text and code require team review before submission.

## Team

- Shramon - Finance methodology and interpretation
- Data and Modelling Lead - data pipeline, model and backtest
- Driscoll - Website, integration, deployment and demonstration

Contribution statements and AI disclosure are included in the final written report appendix.
