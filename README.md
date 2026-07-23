# Event-Based Trading | Team Prosper

This repository contains Team Prosper's FINS5557 Track A project: an AI/ML
finance application that demonstrates how prediction-market probability
features can be converted into short-window equity-index return signals.

The project has two linked parts:

- `ml/`, `models/` and `output/`: data generation, model training, walk-forward
  evaluation and exported handoff artifacts.
- `website/`: Streamlit application used for the final public demo, model
  inference display, methodology explanation and limitations disclosure.

## Final Website

Streamlit entrypoint:

```text
website/app.py
```

The Streamlit dependencies are in:

```text
website/requirements.txt
```

The Website uses the frozen artifacts in `website/data/`, including:

- `model_bundle.pkl`
- `model_metadata.json`
- `simulated_features.csv`
- `prediction_fixture.csv`
- `model_comparison.csv`
- `cumulative_returns.csv`
- `data_quality_summary.csv`

## Current Model Scope

- Selected Website model: `M2_ols`
- Target: `ret_SPY_60m`
- Asset scope: SPY only
- Cost assumption: nil transaction costs and no slippage
- Demo mode: simulated scenario rows, not live Kalshi API data
- Claim boundary: event-window research demonstration, not investment advice

## Local Website Run

```bash
cd website
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run app.py
```

## Public Deployment

Deploy on Streamlit Community Cloud with:

- Repository: `vaibhavjha100/Event-Based-Trading`
- Branch: `main`
- Main file path: `website/app.py`

Streamlit Community Cloud supports an app entrypoint in a subdirectory and can
use the `requirements.txt` file located next to that entrypoint.

## Responsible Use

This application is a research and education prototype. It does not provide
personalised financial advice, does not execute trades and does not collect
personal financial information. AI-assisted coding and drafting must be
disclosed in the final written report.
