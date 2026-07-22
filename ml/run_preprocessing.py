"""Preprocessing pipeline entrypoint.

Usage
-----
python -m ml.run_preprocessing
python -m ml.run_preprocessing --skip-gemini
python -m ml.run_preprocessing --gemini-failed-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.preprocess_config import DEFAULT_PREPROCESS_CONFIG, PreprocessConfig
from ml.preprocessing.align import (
    build_controls,
    build_event_consensus_features,
    split_pm_ohlcv_windows,
)
from ml.preprocessing.band_semantics import build_band_semantics
from ml.preprocessing.contract_map_build import build_contract_event_map
from ml.preprocessing.contract_select import select_universes
from ml.preprocessing.gemini_classify import classify_contracts
from ml.preprocessing.io_utils import write_csv, write_json
from ml.preprocessing.loaders import load_raw_bundle, normalize_contracts
from ml.preprocessing.signals_clean import compute_event_signals
from ml.preprocessing.targets import pivot_primary_targets, recompute_event_returns
from ml.preprocessing.validate_preprocess import run_preprocess_validation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess raw data into cleaned ML-ready tables.")
    p.add_argument("--raw-dir", type=str, default=None)
    p.add_argument("--cleaned-dir", type=str, default=None)
    p.add_argument("--skip-gemini", action="store_true", help="Use rule-based classification only.")
    p.add_argument("--force-gemini", action="store_true", help="Ignore Gemini cache.")
    p.add_argument(
        "--gemini-failed-only",
        action="store_true",
        help="Only re-call Gemini for cache misses/errors.",
    )
    return p.parse_args()


def run_preprocessing(cfg: PreprocessConfig) -> dict:
    t0 = time.time()
    out = Path(cfg.cleaned_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    print("[1] Load + standardize raw data")
    bundle = load_raw_bundle(cfg.raw_dir)
    events = bundle.events.copy()
    consensus = bundle.consensus.copy()
    write_csv(events, out / "events_cleaned.csv")
    write_csv(consensus, out / "consensus_cleaned.csv")
    write_csv(bundle.kalshi, out / "kalshi_contracts_cleaned.csv")
    write_csv(bundle.polymarket, out / "polymarket_contracts_cleaned.csv")
    write_csv(bundle.band_probs, out / "band_probs_cleaned.csv")
    write_csv(bundle.vix, out / "vix_cleaned.csv")
    write_csv(bundle.daily_equity, out / "daily_equity_cleaned.csv")

    contracts_norm = normalize_contracts(bundle.kalshi, bundle.polymarket)
    write_csv(contracts_norm, out / "contracts_normalized.csv")

    print("[2] Gemini / rule classification + band extraction")
    classifications = classify_contracts(bundle.kalshi, bundle.polymarket, cfg)
    write_csv(classifications, out / "contract_classifications.csv")

    print("[3] Contract-event map (optional raw or rebuilt)")
    cmap = build_contract_event_map(
        classifications, bundle.kalshi, bundle.polymarket, bundle.contract_event_map
    )
    write_csv(cmap, out / "contract_event_map_cleaned.csv")

    print("[4] Universe selection")
    primary, research, exclusions = select_universes(classifications, cmap, cfg)
    write_csv(primary, out / "universe_primary_contracts.csv")
    write_csv(research, out / "universe_research_contracts.csv")
    write_csv(exclusions, out / "exclusions_audit.csv")

    print("[5] Band semantics")
    band_sem = build_band_semantics(
        events, consensus, classifications, bundle.band_probs, primary, research
    )
    write_csv(band_sem, out / "band_semantics_cleaned.csv")

    print("[6] PM OHLCV window split (pre vs reaction)")
    pre_ohlcv, rx_ohlcv = split_pm_ohlcv_windows(
        bundle.kalshi_ohlcv,
        bundle.poly_ohlcv,
        events,
        bundle.kalshi,
        bundle.polymarket,
    )
    write_csv(pre_ohlcv, out / "pm_ohlcv_prerelease.csv")
    write_csv(rx_ohlcv, out / "pm_ohlcv_reaction.csv")

    print("[7] Leakage-safe signals")
    signals, signals_meta = compute_event_signals(
        events,
        bundle.band_probs,
        band_sem,
        bundle.kalshi_ohlcv,
        bundle.poly_ohlcv,
        bundle.kalshi,
        bundle.polymarket,
        primary,
    )

    print("[8] Consensus features + controls")
    cons_feat = build_event_consensus_features(events, consensus)
    controls = build_controls(events, bundle.vix, bundle.daily_equity)

    print("[9] Recompute returns from equity 1-min (source of truth)")
    returns_clean, crosscheck = recompute_event_returns(
        events,
        bundle.equity_1min,
        ohlcv_post_minutes=cfg.ohlcv_post_minutes,
        raw_returns=bundle.event_returns_raw,
    )
    write_csv(returns_clean, out / "event_returns_cleaned.csv")
    targets_wide = pivot_primary_targets(returns_clean)

    print("[10] Build event_level_ml_base")
    event_ml = (
        events[["event_id", "event_type", "macro_variable", "release_time", "regime", "day_of_week"]]
        .merge(cons_feat, on="event_id", how="left")
        .merge(signals, on="event_id", how="left")
        .merge(controls.drop(columns=["regime", "event_type", "release_time"], errors="ignore"), on="event_id", how="left")
        .merge(targets_wide, on="event_id", how="left")
    )
    # Prefer surprise/consensus from cons_feat naming
    write_csv(event_ml, out / "event_level_ml_base.csv")

    print("[11] Validation + report")
    report = run_preprocess_validation(
        cfg=cfg,
        raw_counts=bundle.row_counts_raw,
        classifications=classifications,
        primary=primary,
        research=research,
        exclusions=exclusions,
        band_probs=bundle.band_probs,
        band_semantics=band_sem,
        signals_meta=signals_meta,
        event_ml=event_ml,
        returns_crosscheck=crosscheck,
        events=events,
    )
    report["elapsed_seconds"] = round(time.time() - t0, 2)
    report["gemini_enabled"] = cfg.gemini_enabled
    write_json(report, out / "preprocessing_report.json")

    for msg in report["messages"]:
        print(f"  {msg}")
    print(f"Done in {report['elapsed_seconds']}s -> {out}")
    return report


def main() -> None:
    args = parse_args()
    kwargs = {
        "gemini_enabled": not args.skip_gemini,
        "gemini_force": args.force_gemini,
        "gemini_failed_only": args.gemini_failed_only,
    }
    if args.raw_dir:
        kwargs["raw_dir"] = Path(args.raw_dir)
    if args.cleaned_dir:
        kwargs["cleaned_dir"] = Path(args.cleaned_dir)
    cfg = DEFAULT_PREPROCESS_CONFIG.with_overrides(**kwargs)
    report = run_preprocessing(cfg)
    if not report.get("passed", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
