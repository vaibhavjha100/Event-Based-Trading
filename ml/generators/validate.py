"""Validation checks and generation manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml.schemas import SCHEMA_BY_FILE
from ml.synth_config import WINDOW_MINUTES, SynthConfig


def run_validations(
    *,
    events: pd.DataFrame,
    latents: pd.DataFrame,
    band_probs: pd.DataFrame,
    signals: pd.DataFrame,
    event_returns: pd.DataFrame,
    equity_1min: pd.DataFrame,
    daily_equity: pd.DataFrame,
    contract_map: pd.DataFrame,
    kalshi: pd.DataFrame,
    polymarket: pd.DataFrame,
    consensus: pd.DataFrame,
    cfg: SynthConfig,
) -> List[str]:
    """Run sanity checks; return list of human-readable messages (incl. failures)."""
    messages: List[str] = []
    errors: List[str] = []

    # 1) Probabilities sum to 1
    if not band_probs.empty:
        grp = band_probs.groupby(["event_id", "platform", "timestamp"])["probability"].sum()
        max_dev = float((grp - 1.0).abs().max())
        messages.append(f"prob_sum max deviation from 1: {max_dev:.2e}")
        if max_dev > 1e-5:
            errors.append(f"Probabilities do not sum to 1 (max |sum-1|={max_dev})")

    # 2) PMPS_pre only from pre-release windows (structural check on minutes used)
    # Recompute from band_probs and compare
    if not band_probs.empty and not signals.empty:
        recomputed = []
        for eid in signals["event_id"]:
            sub = band_probs[band_probs["event_id"] == eid]
            def pbad(platform, mins):
                g = sub[(sub["platform"] == platform) & (sub["minutes_to_release"] == mins)]
                if g.empty:
                    return np.nan
                return float(g.loc[g["is_adverse"], "probability"].sum())
            recomputed.append(
                {
                    "event_id": eid,
                    "PMPS_pre_K": pbad("KALSHI", -5) - pbad("KALSHI", -60),
                    "PMPS_pre_P": pbad("POLYMARKET", -5) - pbad("POLYMARKET", -60),
                }
            )
        rec = pd.DataFrame(recomputed).dropna()
        merged = signals.merge(rec, on="event_id", suffixes=("", "_re"))
        if len(merged):
            dk = float((merged["PMPS_pre_K"] - merged["PMPS_pre_K_re"]).abs().max())
            dp = float((merged["PMPS_pre_P"] - merged["PMPS_pre_P_re"]).abs().max())
            messages.append(f"PMPS_pre recompute max |d| K={dk:.2e} P={dp:.2e}")
            if dk > 1e-8 or dp > 1e-8:
                errors.append("PMPS_pre does not match pre-release band probabilities")

    # 3) prior_5d_SPY_return consistency with daily_equity
    spy = daily_equity[daily_equity["asset"] == "SPY"].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy["prior_5d"] = np.log(spy["close"] / spy["close"].shift(5))
    max_prior_err = 0.0
    for _, ev in events.iterrows():
        eid = int(ev["event_id"])
        release = pd.Timestamp(ev["release_time"])
        day = release.tz_localize(None).normalize() if release.tzinfo else release.normalize()
        prev = spy[spy["date"] < day]
        if len(prev) == 0:
            continue
        expected = prev["prior_5d"].iloc[-1]
        if np.isnan(expected):
            continue
        got = float(signals.loc[signals["event_id"] == eid, "prior_5d_SPY_return"].iloc[0])
        max_prior_err = max(max_prior_err, abs(got - expected))
    messages.append(f"prior_5d_SPY_return max |d| vs daily: {max_prior_err:.2e}")
    if max_prior_err > 1e-6:
        errors.append("prior_5d_SPY_return inconsistent with daily_equity_prices")

    # 4) Event-window returns vs equity_1min
    if not equity_1min.empty and not event_returns.empty:
        max_ret_err = _check_return_consistency(events, event_returns, equity_1min, cfg)
        messages.append(f"event_returns vs 1min recompute max |d|: {max_ret_err:.2e}")
        if max_ret_err > 5e-3:
            errors.append(
                f"equity_1min paths inconsistent with event_returns (max |d|={max_ret_err})"
            )

    # 5) contract_event_map covers all core macro contracts
    core_k = set(
        kalshi.loc[kalshi["edge_case_type"] == "none", "ticker"].astype(str)
    )
    core_p = set(
        polymarket.loc[polymarket["edge_case_type"] == "none", "condition_id"].astype(str)
    )
    map_k = set(
        contract_map.loc[
            (contract_map["platform"] == "KALSHI")
            & (contract_map["is_core_macro_contract"]),
            "contract_id",
        ].astype(str)
    )
    map_p = set(
        contract_map.loc[
            (contract_map["platform"] == "POLYMARKET")
            & (contract_map["is_core_macro_contract"]),
            "contract_id",
        ].astype(str)
    )
    missing_k = core_k - map_k
    missing_p = core_p - map_p
    messages.append(
        f"contract_map core coverage: Kalshi missing={len(missing_k)} Poly missing={len(missing_p)}"
    )
    if missing_k or missing_p:
        errors.append(
            f"contract_event_map missing core contracts K={len(missing_k)} P={len(missing_p)}"
        )

    # Correlations (informational)
    m = events.merge(latents, on="event_id").merge(
        signals[["event_id", "PMPS_pre_K", "surprise_value"]], on="event_id"
    )
    spy60 = event_returns[
        (event_returns["asset"] == "SPY") & (event_returns["window"] == "60m")
    ][["event_id", "return_value"]].rename(columns={"return_value": "R_SPY_60m"})
    m = m.merge(spy60, on="event_id", how="left")
    # Only macro events with band probs
    macro = m[m["event_type"] != "NOISE"]
    if len(macro) > 5:
        c1 = float(macro["Z_e"].corr(macro["surprise_value"]))
        c2 = float(macro["Z_e"].corr(macro["PMPS_pre_K"]))
        c3 = float(macro["PMPS_pre_K"].corr(macro["R_SPY_60m"]))
        messages.append(
            f"correlations: corr(Z,Surprise)={c1:.3f} corr(Z,PMPS_K)={c2:.3f} "
            f"corr(PMPS_K,R_SPY_60m)={c3:.3f}"
        )

    # Ranges
    messages.append(
        f"events={len(events)} band_prob_rows={len(band_probs)} "
        f"signals={len(signals)} returns={len(event_returns)}"
    )

    if errors:
        for e in errors:
            messages.append(f"ERROR: {e}")
    else:
        messages.append("All validation checks passed.")

    return messages


def _check_return_consistency(
    events: pd.DataFrame,
    event_returns: pd.DataFrame,
    equity_1min: pd.DataFrame,
    cfg: SynthConfig,
) -> float:
    eq = equity_1min.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, format="ISO8601")
    max_err = 0.0
    release_map = {
        int(r.event_id): pd.Timestamp(r.release_time) for r in events.itertuples()
    }

    # Sample up to 40 event/asset pairs for speed
    sample = event_returns[
        (event_returns["window"].isin(["5m", "30m", "60m"]))
    ].drop_duplicates(["event_id", "asset"]).head(40)

    for _, row in sample.iterrows():
        eid = int(row["event_id"])
        asset = row["asset"]
        release = release_map[eid]
        if release.tzinfo is None:
            release = release.tz_localize("UTC")
        else:
            release = release.tz_convert("UTC")
        t0m5 = release - pd.Timedelta(minutes=5)
        win = eq[(eq["asset"] == asset) & (eq["event_id"] == eid)]
        if win.empty:
            continue
        for w, mins in WINDOW_MINUTES.items():
            if w == "EOD":
                continue
            end = release + pd.Timedelta(minutes=mins)
            p0 = win.loc[win["timestamp"] == t0m5, "close"]
            p1 = win.loc[win["timestamp"] == end, "close"]
            if p0.empty or p1.empty:
                continue
            recomputed = float(np.log(p1.iloc[0] / p0.iloc[0]))
            target = float(
                event_returns.loc[
                    (event_returns["event_id"] == eid)
                    & (event_returns["asset"] == asset)
                    & (event_returns["window"] == w),
                    "return_value",
                ].iloc[0]
            )
            max_err = max(max_err, abs(recomputed - target))
    return max_err


def write_manifest(
    out_dir: Path,
    cfg: SynthConfig,
    frames: Dict[str, pd.DataFrame],
    validation_messages: List[str],
) -> Path:
    """Write generation_manifest.json with row counts and schema version."""
    files = {}
    for name, df in frames.items():
        files[name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "expected_schema": SCHEMA_BY_FILE.get(name, []),
        }
    payload: Dict[str, Any] = {
        "schema_version": cfg.schema_version,
        "seed": cfg.seed,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "validation": validation_messages,
    }
    path = out_dir / "generation_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
