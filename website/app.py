from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.data_loader import load_data


DATA_DIR = APP_DIR / "data"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(
    page_title="Team Prosper | Macro Signal Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17212b;
        --muted: #5f6b76;
        --line: #d8dee4;
        --paper: #ffffff;
        --wash: #f5f7f8;
        --green: #087f5b;
        --red: #c92a2a;
        --amber: #b26a00;
        --teal: #0b7285;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #f3f5f6; border-right: 1px solid var(--line); }
    [data-testid="stMetric"] {
        border-top: 3px solid var(--teal);
        padding: 0.7rem 0.8rem;
        background: var(--wash);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
    h1 { font-size: 2.15rem; margin-bottom: 0.25rem; }
    h2 { font-size: 1.35rem; margin-top: 1.6rem; }
    h3 { font-size: 1.05rem; }
    .eyebrow {
        color: var(--teal); font-size: 0.78rem; font-weight: 700;
        text-transform: uppercase; margin-bottom: 0.25rem;
    }
    .subtle { color: var(--muted); font-size: 0.92rem; }
    .evidence {
        border-left: 4px solid var(--teal); background: var(--wash);
        padding: 0.9rem 1rem; margin: 0.5rem 0 1rem 0;
    }
    .risk {
        border-left: 4px solid var(--amber); background: #fff8e6;
        padding: 0.9rem 1rem; margin: 0.5rem 0 1rem 0;
    }
    .status-live { color: var(--green); font-weight: 700; }
    .status-demo { color: var(--amber); font-weight: 700; }
    .small-label { color: var(--muted); font-size: 0.8rem; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); }
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
        visibility: hidden;
        height: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cached_load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, bool, list[str]]:
    return load_data(DATA_DIR)


@st.cache_data
def cached_model_comparison() -> pd.DataFrame:
    path = DATA_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


signals, performance, metrics, is_demo, data_issues = cached_load_data()
model_comparison = cached_model_comparison()


def list_text(values: list[str]) -> str:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    return ", ".join(cleaned) if cleaned else "Not supplied"


def date_window_text(frame: pd.DataFrame, column: str, unit: str) -> str:
    if frame.empty or column not in frame:
        return "Not supplied"
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return "Not supplied"
    return f"{dates.min().date()} to {dates.max().date()} ({len(frame)} {unit})"


asset_text = list_text(sorted(signals["target_asset"].dropna().unique().tolist()))
category_text = list_text(sorted(signals["category"].dropna().unique().tolist()))
evaluation_window = date_window_text(performance, "date", "events")
scenario_window = date_window_text(signals, "date", "scenario rows")
model_scope = (
    f"{metrics.get('model_version', 'Model not supplied')} / "
    f"{metrics.get('target_variable', 'Target not supplied')} / {asset_text}"
)
artifact_source = "Frozen model artifacts plus simulated Kalshi-style scenario rows"
cost_bps = int(metrics.get("transaction_cost_bps", 0))
cost_note = (
    "No transaction costs included"
    if cost_bps == 0
    else f"{cost_bps} bps per trade included"
)
test_return_label = "Test return" if cost_bps == 0 else "Net test return"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtle">{description}</div>', unsafe_allow_html=True)


def provenance() -> None:
    status = "DEMO FALLBACK" if is_demo else "FROZEN ARTIFACTS"
    css = "status-demo" if is_demo else "status-live"
    st.divider()
    st.markdown(
        f"""
        <span class="small-label">Data status:</span> <span class="{css}">{status}</span>
        &nbsp;&nbsp; <span class="small-label">Model:</span> {metrics.get('model_version', 'Not supplied')}
        &nbsp;&nbsp; <span class="small-label">Updated:</span> {metrics.get('data_updated_at', 'Not supplied')}
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("### Team Prosper")
    st.caption("Prediction Market Macro Signal Lab")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Scenario Signals",
            "Backtest",
            "Agent Pipeline",
            "Methodology",
            "Data & Limitations",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Evaluation window")
    st.write(evaluation_window)
    st.caption("Scenario rows")
    st.write(scenario_window)
    st.caption("Model / target")
    st.write(model_scope)
    st.caption("Target assets")
    st.write(asset_text)
    st.caption("Primary source")
    st.write(artifact_source)
    if is_demo:
        st.warning("Demo outputs are shown until validated team files are supplied.")
        with st.expander("Data contract status"):
            for issue in data_issues:
                st.write(f"- {issue}")


if page == "Overview":
    page_header(
        "Frozen model demonstration",
        "Do prediction-market features improve event-window equity forecasts?",
        f"The website runs inference from the exported {metrics.get('model_version', 'model')} artifact on {metrics.get('target_variable', 'the exported target')}. It displays {asset_text} output and the exported scenario categories ({category_text}) using simulated rows for the website demo.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(test_return_label, fmt_pct(metrics["strategy_return"]), f"{(metrics['strategy_return'] - metrics['benchmark_return']) * 100:.1f} pp vs benchmark")
    c2.metric("Test Sharpe", f"{metrics['sharpe']:.2f}", cost_note)
    c3.metric("Max drawdown", fmt_pct(metrics["max_drawdown"]), "Held-out period")
    c4.metric("Directional accuracy", fmt_pct(metrics["test_directional_accuracy"]), f"n = {metrics['number_of_trades']} trades")

    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Exported walk-forward performance")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=performance["date"], y=performance["strategy"], name="Strategy", line=dict(color="#087f5b", width=2.5)))
        fig.add_trace(go.Scatter(x=performance["date"], y=performance["benchmark"], name="Benchmark", line=dict(color="#6c757d", width=2)))
        fig.update_layout(yaxis_title="Growth of 100", xaxis_title=None, legend_orientation="h", margin=dict(l=10, r=10, t=10, b=10), height=410)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with right:
        st.subheader("Research claim")
        st.markdown(
            """
            <div class="evidence">
            <b>Primary hypothesis</b><br>
            Prediction-market probability features and macro surprise are tested as predictors of short-window SPY event returns. The incremental test compares the selected M2 OLS model with the M0 consensus-surprise baseline using the frozen walk-forward artifact.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Key tests")
        st.write("1. M0 consensus surprise baseline vs M2 prediction-market distribution features")
        st.write("2. Event-window forecast using the exported `ret_SPY_60m` target")
        st.write("3. Walk-forward directional accuracy, clipped Sharpe-like score and clipped R-squared")
        st.write(f"4. Strategy output under the exported cost assumption: {cost_note.lower()}")

    st.subheader("Research design")
    design = pd.DataFrame(
        [
            ["Training", metrics.get("train_period", "Not supplied"), "Estimate model parameters"],
            ["Validation", metrics.get("validation_period", "Not supplied"), "Select fixed hyperparameters without using test outcomes"],
            ["Testing", evaluation_window, "Walk-forward evaluation artifact"],
            ["Scenario inference", scenario_window, "Website-side simulated input demonstration"],
        ],
        columns=["Split", "Exported period", "Permitted use"],
    )
    st.dataframe(design, width="stretch", hide_index=True)
    provenance()


elif page == "Scenario Signals":
    page_header(
        "Scenario model output",
        "Simulated event-window scenario inputs",
        "This page demonstrates Website-side inference using model-ready simulated rows. The current model includes macro surprise, so the output is an event-window scenario result, not a pre-release live trading signal.",
    )
    latest_date = signals["date"].max()
    latest = signals.loc[signals["date"] == latest_date].copy()
    f1, f2, f3 = st.columns(3)
    categories = f1.multiselect("Category", sorted(latest["category"].unique()), default=sorted(latest["category"].unique()))
    assets = f2.multiselect("Target asset", sorted(latest["target_asset"].unique()), default=sorted(latest["target_asset"].unique()))
    directions = f3.multiselect("Direction", ["Long", "Neutral", "Short"], default=["Long", "Neutral", "Short"])
    view = latest[
        latest["category"].isin(categories)
        & latest["target_asset"].isin(assets)
        & latest["signal_direction"].isin(directions)
    ].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("As of", str(latest_date.date()))
    c2.metric("Markets monitored", len(view))
    c3.metric("Risk-off signals", int((view["signal_direction"] == "Short").sum()))
    c4.metric("Median confidence", fmt_pct(view["classification_confidence"].median()) if len(view) else "N/A")

    display = view[
        ["market_title", "category", "target_asset", "p_bad", "pmps_pre", "volume", "open_interest", "signal_direction", "model_forecast"]
    ].copy()
    display["p_bad"] = display["p_bad"].map(lambda x: fmt_pct(x))
    display["pmps_pre"] = display["pmps_pre"].map(lambda x: fmt_pct(x))
    display["model_forecast"] = display["model_forecast"].map(lambda x: fmt_pct(x, 2))
    st.dataframe(display, width="stretch", hide_index=True)

    if not view.empty:
        selected_title = st.selectbox("Inspect market", view["market_title"].tolist())
        history = signals.loc[signals["market_title"] == selected_title].sort_values("date")
        selected = view.loc[view["market_title"] == selected_title].iloc[0]
        left, right = st.columns([1.6, 1])
        with left:
            pmps_abs_max = pd.to_numeric(history["pmps_pre"], errors="coerce").abs().max()
            pmps_axis_max = max(float(pmps_abs_max) * 1.35, 0.025)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=history["date"],
                    y=history["p_bad"],
                    name="Adverse probability",
                    mode="lines+markers" if len(history) > 1 else "markers",
                    marker=dict(color="#0b7285", size=9),
                    line=dict(color="#0b7285", width=2.5),
                )
            )
            fig.add_trace(
                go.Bar(
                    x=history["date"],
                    y=history["pmps_pre"],
                    name="PMPS pre-event",
                    marker_color="#d9480f",
                    yaxis="y2",
                    opacity=0.42,
                    width=None if len(history) > 1 else [12 * 60 * 60 * 1000],
                )
            )
            fig.update_layout(
                yaxis=dict(title="Adverse probability", tickformat=".0%", range=[0, 1]),
                yaxis2=dict(title="Daily change", tickformat=".1%", range=[-pmps_axis_max, pmps_axis_max], overlaying="y", side="right", showgrid=False),
                legend_orientation="h", margin=dict(l=10, r=10, t=25, b=10), height=400,
            )
            if len(history) == 1:
                center = pd.to_datetime(history["date"]).iloc[0]
                fig.update_xaxes(
                    range=[center - pd.Timedelta(days=3), center + pd.Timedelta(days=3)],
                    tickformat="%b %d, %Y",
                )
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        with right:
            st.subheader("Signal audit")
            st.write(f"Direction: **{selected['signal_direction']} {selected['target_asset']}**")
            st.write(f"Adverse probability: **{fmt_pct(selected['p_bad'])}**")
            st.write(f"PMPS pre-event: **{fmt_pct(selected['pmps_pre'])}**")
            st.write(f"Scenario timestamp: `{selected['known_at']}`")
            st.write(f"Classification confidence: **{fmt_pct(selected['classification_confidence'])}**")
            st.caption("Scenario rows are simulated for demonstration. Data confirmed that surprise is not available at the present release decision time, so this model should be described as event-window inference.")
    provenance()


elif page == "Backtest":
    page_header(
        "Frozen evaluation artifact",
        "Performance, robustness and model-selection evidence",
        "The backtest reports statistical fit, directional accuracy and strategy output under the exported nil-cost assumption, using the chronological walk-forward file supplied by the Data Lead.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(test_return_label, fmt_pct(metrics["strategy_return"]))
    c2.metric("Sharpe ratio", f"{metrics['sharpe']:.2f}")
    c3.metric("Hit rate", fmt_pct(metrics["hit_rate"]))
    c4.metric("Transaction cost", f"{cost_bps} bps / trade")

    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Drawdown", "Model comparison", "Ablation plan"])
    with tab1:
        fig = px.line(performance, x="date", y=["strategy", "benchmark"], color_discrete_map={"strategy": "#087f5b", "benchmark": "#6c757d"})
        fig.update_layout(yaxis_title="Growth of 100", xaxis_title=None, legend_title=None, legend_orientation="h", height=440)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with tab2:
        fig = px.area(performance, x="date", y="drawdown", color_discrete_sequence=["#c92a2a"])
        fig.update_layout(yaxis_title="Drawdown", yaxis_tickformat=".0%", xaxis_title=None, height=440)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with tab3:
        evidence = pd.DataFrame(
            [
                [metrics.get("selected_model_id", metrics.get("model_version", "Selected model")), metrics.get("target_variable", "Post-event 60m return"), metrics["test_r2"], "Walk-forward continuous forecast"],
                ["Directional score", "Positive vs non-positive return", metrics["test_directional_accuracy"], "Sign accuracy from the regression forecast"],
                ["Trading rule", "Long / neutral / short", metrics["hit_rate"], f"Economic implementation; {cost_note.lower()}"],
            ],
            columns=["Specification", "Target", "Held-out score", "Purpose"],
        )
        st.dataframe(evidence, width="stretch", hide_index=True)
        if not model_comparison.empty:
            st.subheader("Exported model-selection table")
            preferred_cols = [
                "model_id",
                "config_id",
                "estimator_name",
                "selection_score",
                "directional_accuracy",
                "sharpe_like",
                "r2",
                "rmse",
                "cumulative_return",
                "n_trades",
                "selected",
            ]
            available_cols = [column for column in preferred_cols if column in model_comparison.columns]
            comparison_view = model_comparison[available_cols].copy()
            for column in [
                "selection_score",
                "directional_accuracy",
                "sharpe_like",
                "r2",
                "rmse",
                "cumulative_return",
            ]:
                if column in comparison_view.columns:
                    comparison_view[column] = pd.to_numeric(comparison_view[column], errors="coerce").round(4)
            if "selected" in comparison_view.columns:
                comparison_view["selected"] = comparison_view["selected"].map(
                    lambda flag: "Selected" if str(flag).lower() == "true" else ""
                )
            if "selection_score" in comparison_view.columns:
                comparison_view = comparison_view.sort_values("selection_score", ascending=False)
            st.dataframe(comparison_view, width="stretch", hide_index=True)
            st.caption("M2_ols is the frozen website model. M3_gbdt scores higher in the export but is not the chosen artifact, so the report should explain the stability and interpretability reason for using M2_ols.")
        st.markdown(
            """
            <div class="risk"><b>Interpretation rule</b><br>
            The website follows the frozen M2_ols handoff. Because surprise is not available before the release, the model should be described as event-window evidence rather than a fully pre-release trading strategy.</div>
            """,
            unsafe_allow_html=True,
        )
    with tab4:
        ablation = pd.DataFrame(
            [
                ["M0", "Consensus surprise + controls", "Textbook baseline", "Core"],
                ["M1", "+ pre-event PMPS", "Incremental prediction-market value", "Core"],
                ["M2", "+ distribution moments", "Selected model family: mean, variance and skew repricing", "Current"],
                ["M3", "+ liquidity weighting", "Information-backed conviction", "Robustness"],
                ["M4", "+ regime interactions", "Asymmetric and state-dependent response", "Extension"],
                ["M5", "+ cross-platform disagreement", "Independent-pool uncertainty", "Bonus only"],
            ],
            columns=["Model", "Adds", "Question answered", "Scope"],
        )
        st.dataframe(ablation, width="stretch", hide_index=True)
        st.caption("Each model must report held-out R-squared, RMSE and directional accuracy. Test performance is never used to select the main specification.")
    provenance()


elif page == "Agent Pipeline":
    page_header(
        "Auditable multi-stage system",
        "From prediction-market contract to equity decision",
        "The language agent handles semantic interpretation; deterministic code handles probability, timing, modelling and performance calculations.",
    )
    stages = [
        ("1. Collect", "Kalshi metadata and trades", "Timestamped raw market tables", "Data Lead"),
        ("2. Classify", "Title, subtitle and contract rules", "Category, event date, threshold, YES-is-adverse, confidence", "Finance + Agent"),
        ("3. Construct", "YES price, volume and open interest", "p_bad, PMPS_pre, PMPS_reaction and distribution moments", "Data Lead"),
        ("4. Match", "Release timestamp, consensus and ETF prices", "Non-overlapping 5m/30m/60m/EOD returns and controls", "Finance + Data"),
        ("5. Train", "Chronological event sample", "Walk-forward OLS/Elastic Net and frozen thresholds", "Data Lead"),
        ("6. Evaluate", "Locked post-July 2023 sample", "Regression, classification and backtest evidence", "All"),
        ("7. Publish", "Frozen model bundle + CSV/JSON artifacts", "Website inference, methodology and reproducibility record", "Website Lead"),
    ]
    st.dataframe(pd.DataFrame(stages, columns=["Stage", "Input", "Output", "Owner"]), width="stretch", hide_index=True)
    st.subheader("Agent guardrails")
    g1, g2, g3 = st.columns(3)
    g1.info("Structured JSON output with a fixed schema and confidence score.")
    g2.info("Low-confidence or ambiguous contracts are reviewed or excluded.")
    g3.info("The language agent classifies contracts; tested Python code calculates features, returns and models; the Website only runs inference from the frozen model.")
    st.subheader("Required classification record")
    st.code(
        """{
  "category": "CPI",
  "economic_variable": "headline_cpi_yoy",
  "event_date": "2024-12-11",
  "threshold": 3.0,
  "yes_is_adverse": true,
  "equity_direction": "negative",
  "confidence": 0.96,
  "reason": "Higher inflation implies tighter policy expectations",
  "include": true
}""",
        language="json",
    )
    provenance()


elif page == "Methodology":
    page_header(
        "Event-window financial logic",
        "Hypotheses, variables and identification",
        "The Data Lead confirmed that macro surprise is unavailable at the present release decision time. The current website therefore presents an event-window model, with simulated rows used only for inference demonstration.",
    )
    st.subheader("Hypotheses")
    hypotheses = pd.DataFrame(
        [
            ["H1", f"Prediction-market features and macro surprise forecast short-window {asset_text} event returns.", "Strong held-out directional accuracy on ret_SPY_60m"],
            ["H2", "Prediction-market distribution features add information beyond Actual minus Consensus.", "M2 improves the Data Lead selection score relative to weaker alternatives"],
            ["H3", "Event reactions are asymmetric across surprise direction and monetary regime.", "Direction/regime interaction improves held-out fit"],
            ["H4", "Model selection should balance accuracy, economic score and fit.", "Selection score uses directional accuracy, clipped Sharpe-like score and clipped R-squared"],
        ],
        columns=["ID", "Testable statement", "Expected evidence"],
    )
    st.dataframe(hypotheses, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Signal definitions")
        st.latex(r"p^{bad}_{e,t} = p^{YES}_{e,t}\;\text{if YES is adverse, else}\;1-p^{YES}_{e,t}")
        st.latex(r"PMPS^{pre}_{e}=p^{bad}_{e,t_0-5m}-p^{bad}_{e,t_0-60m}")
        st.latex(r"Surprise_e = Actual_e - Consensus_e")
        st.latex(r"R^{window}_{e,a}(w)=\log(P_{a,t_0+w}/P_{a,t_0-5m}),\quad w=60m")
        st.caption("The current primary target is ret_SPY_60m. Surprise enters the model, so the inference scope is event-window analysis.")
    with right:
        st.subheader("Timing boundary")
        st.latex(r"PMPS^{reaction}_{e}=p^{bad}_{e,t_0+30m}-p^{bad}_{e,t_0-30m}")
        st.write("PMPS_pre is observed before release, but `surprise` is observed only at or after release. The displayed model should not be described as a pre-release decision model.")
        st.caption("The limitation is disclosed in the website because it materially changes the trading interpretation.")

    st.subheader("Identification and controls")
    i1, i2 = st.columns(2)
    with i1:
        st.write("- Chronological train, validation and test split")
        st.write("- Feature timing is disclosed for every model family")
        st.write("- Market result excluded from all predictors")
        st.write("- Expanding-window re-estimation for each next event")
    with i2:
        st.write("- Actual minus Consensus baseline")
        st.write("- VIX at T-1 and prior five-day index return")
        st.write("- Exported event-type and day-of-week indicators")
        st.write("- Contract clustering or event-level aggregation")

    st.subheader("Model ladder")
    st.write(f"Current website handoff displays {metrics.get('model_version', 'the exported model')} on {metrics.get('target_variable', 'the exported target')}. Data confirmed the selection score combines directional accuracy, clipped Sharpe-like performance and clipped R-squared; M2_ols is treated as the frozen main specification for interpretability and consistency with the finance methodology.")
    ladder = pd.DataFrame(
        [
            ["M0", "OLS: consensus surprise + controls", "Required baseline"],
            ["M1", "OLS: M0 + PMPS_pre", "Primary incremental-value test"],
            ["M2", "OLS: M1 + distribution moments", "Current selected model family"],
            ["M3", "Liquidity-weighted features with OLS / Elastic Net / GBDT variants", "Robustness extension"],
        ],
        columns=["Model", "Method", "Role"],
    )
    st.dataframe(ladder, width="stretch", hide_index=True)
    st.markdown(
        "Evidence base: [Diercks, Katz & Wright (2026)](https://www.federalreserve.gov/econres/feds/kalshi-and-the-rise-of-macro-markets.htm) validates Kalshi as a distributionally rich real-time expectations source; MacKinlay (1997) motivates the event-study baseline."
    )
    provenance()


else:
    page_header(
        "Evidence quality and scope",
        "Data coverage, reproducibility and limitations",
        "This page makes the boundaries of the evidence visible so users can distinguish a robust finding from an attractive story.",
    )
    data_tab, ethics_tab, testing_tab = st.tabs(["Data & QA", "Ethics & Governance", "User testing"])
    with data_tab:
        st.subheader("Current handoff coverage")
        handoff = pd.DataFrame(
            [
                ["Model used by website", metrics.get("model_version", "Not supplied")],
                ["Confirmed selected model", metrics.get("data_selected_model_id", "Not supplied")],
                ["Target", metrics.get("target_variable", "Not supplied")],
                ["Target asset displayed", asset_text],
                ["Scenario categories", category_text],
                ["Training events", metrics.get("n_train_events", "Not supplied")],
                ["Test events", metrics.get("n_test_events", "Not supplied")],
                ["Model artifacts scored", metrics.get("n_model_artifacts", "Not supplied")],
                ["Evaluation window", evaluation_window],
                ["Scenario rows", scenario_window],
                ["Walk-forward mode", metrics.get("walkforward_mode", "Not supplied")],
                ["Entry / exit", metrics.get("entry_exit", "Not supplied")],
                ["Cost assumption", metrics.get("transaction_cost_note", cost_note)],
                ["Surprise timing", metrics.get("surprise_timing", "Not supplied")],
                ["Date-range note", metrics.get("date_range_note", "Not supplied")],
            ],
            columns=["Item", "Value"],
        )
        handoff["Value"] = handoff["Value"].astype(str)
        st.dataframe(handoff, width="stretch", hide_index=True)

        if metrics.get("selection_reason"):
            st.caption(metrics["selection_reason"])

        st.subheader("Target data contract")
        coverage = pd.DataFrame(
            [
                ["Prediction markets", "Kalshi / Polymarket-style probability features", "Frozen evaluation artifact plus simulated scenario rows", "Contract rules, timestamped probability, volume and OI"],
                ["Macro releases", "Official release + consensus source", category_text, "Release time, actual, consensus and surprise"],
                ["Equity prices", "Approved intraday source", asset_text, "Timestamped index ETF bars matched to the event window"],
                ["Controls", "Approved market source", "Historical walk-forward sample", "VIX T-1, prior five-day return, regime and event type"],
                ["Inference inputs", "Frozen walk-forward pipeline", "Train/validation/test", "Model bundle, feature panel, version and known-at time"],
            ],
            columns=["Dataset", "Source", "Coverage", "Required fields"],
        )
        st.dataframe(coverage, width="stretch", hide_index=True)

        st.subheader("Quality controls")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Train events", metrics.get("n_train_events", "Pending"))
        q2.metric("Test events", metrics.get("n_test_events", "Pending"))
        q3.metric("Model variants", metrics.get("n_model_artifacts", "Pending"))
        q4.metric("Transaction costs", f"{cost_bps} bps")

        st.subheader("Material limitations")
        limitations = pd.DataFrame(
            [
                ["Small event sample", "CPI and FOMC releases are infrequent.", "Use parsimonious models, walk-forward tests and uncertainty intervals."],
                ["Liquidity", "Thin contracts can produce noisy probabilities.", "Apply volume/OI QA and report filtered sensitivity."],
                ["Contract heterogeneity", "Thresholds and settlement rules differ.", "Normalise adverse polarity and aggregate at event level."],
                ["Surprise timing", "Macro surprise is not available before the release decision time.", "Describe the current model as event-window inference, not a pre-release live trading signal."],
                ["Multiple testing", "Many windows and layers invite cherry-picking.", "Pre-specify 60m as primary and label extensions."],
                ["Trading frictions", "The exported strategy uses no transaction costs or slippage.", "Label the result as nil-cost and avoid claiming after-cost profitability."],
                ["Date range", "Data confirmed the simulated-data demo is no longer constrained by the earlier range.", "Disclose the exported walk-forward window and the simulated-row purpose."],
                ["Asset scope", "Data confirmed SPY-only output.", "Do not claim QQQ evidence unless a separate rerun is supplied."],
                ["Model selection", "M3_gbdt has the highest exported selection score while M2_ols is the frozen website model.", "Explain that the main specification favours OLS interpretability and finance-method consistency; show the comparison table instead of hiding the trade-off."],
            ],
            columns=["Risk", "Why it matters", "Mitigation / disclosure"],
        )
        st.dataframe(limitations, width="stretch", hide_index=True)

    with ethics_tab:
        ethics = pd.DataFrame(
            [
                ["Bias", "Coverage and liquidity favour popular contracts and regimes.", "Publish the sample funnel and excluded-market profile."],
                ["Privacy", "The project uses public market and macro data, not personal records.", "Collect no user financial data and log no secrets."],
                ["Responsible AI", "Automated polarity errors could reverse a signal.", "Use confidence thresholds, human review and an error audit."],
                ["Security", "API credentials could leak through the repository.", "Use environment variables and provide `.env.example` only."],
                ["Market impact", "Widespread signal use may create reflexivity or manipulation incentives.", "Provide research-only framing and no automated execution."],
                ["Regulation", "Event-contract rules and platform access can change.", "Cite primary CFTC material and avoid legal or trading advice."],
            ],
            columns=["Dimension", "Project-specific risk", "Design response"],
        )
        st.dataframe(ethics, width="stretch", hide_index=True)
        st.info("Research prototype only. The application does not provide personalised financial advice or place trades.")

    with testing_tab:
        st.write("The marking guide requires at least three test users and a structured feedback instrument. Each tester should complete the same five tasks before rating the application.")
        tasks = pd.DataFrame(
            [
                ["T1", "State the primary hypothesis from Overview", "Correct in under 30 seconds"],
                ["T2", "Identify one current signal and its known-at timestamp", "Correct market, direction and time"],
                ["T3", "Explain why M1 is compared with M0", "Mentions incremental value over consensus"],
                ["T4", "Find the maximum drawdown and transaction-cost assumption", "Both values correct"],
                ["T5", "Name one limitation and mitigation", "Matched risk-response pair"],
            ],
            columns=["Task", "Prompt", "Success criterion"],
        )
        st.dataframe(tasks, width="stretch", hide_index=True)
        test_template = pd.DataFrame(
            columns=["tester_id", "role", "task_id", "success", "time_seconds", "ease_1_to_5", "confidence_1_to_5", "issue", "suggestion"]
        )
        st.download_button(
            "Download testing template",
            test_template.to_csv(index=False),
            file_name="user_testing_template.csv",
            mime="text/csv",
        )
        st.caption("Final evidence must report task success, median completion time, ratings, recurring issues and the design changes made in response.")

    st.markdown(
        """
        <div class="risk"><b>Claim boundary</b><br>
        The project tests whether prediction-market repricing contains incremental short-horizon information. It does not claim causal identification or guaranteed future profitability.</div>
        """,
        unsafe_allow_html=True,
    )
    provenance()
