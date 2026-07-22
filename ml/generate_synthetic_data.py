"""Synthetic prediction-market and equity dataset generator.

Usage
-----
python -m ml.generate_synthetic_data
python -m ml.generate_synthetic_data --seed 7 --start 2021-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as script or module
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.generators.bands import generate_band_probabilities
from ml.generators.consensus import generate_consensus
from ml.generators.contract_map import build_contract_event_map
from ml.generators.equities import (
    generate_daily_equity_prices,
    generate_equity_1min_ohlcv,
    generate_event_returns,
)
from ml.generators.events import generate_events_and_latents
from ml.generators.kalshi import generate_kalshi_contracts, generate_kalshi_ohlcv
from ml.generators.polymarket import (
    generate_polymarket_contracts,
    generate_polymarket_ohlcv,
)
from ml.generators.signals import generate_event_signals
from ml.generators.validate import run_validations, write_manifest
from ml.generators.vix import generate_vix_daily
from ml.synth_config import DEFAULT_CONFIG, SynthConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic event-based trading datasets.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default from config).")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD.")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: data/raw).",
    )
    return p.parse_args()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path.name}: {len(df):,} rows")


def generate_all(cfg: SynthConfig) -> dict:
    """Run the full one-way generation pipeline and write CSVs."""
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/11] Events + latents (seed={cfg.seed}, {cfg.start_date} -> {cfg.end_date})")
    events, latents = generate_events_and_latents(cfg, rng)
    _write_csv(events, out / "events.csv")
    _write_csv(latents, out / "events_latent.csv")

    print("[2/11] Consensus forecasts + Surprise")
    consensus = generate_consensus(events, latents, cfg, rng)
    _write_csv(consensus, out / "consensus_forecasts.csv")

    print("[3/11] Band-level probabilities")
    band_probs = generate_band_probabilities(events, latents, consensus, cfg, rng)
    _write_csv(band_probs, out / "event_band_probabilities.csv")

    print("[4/11] Kalshi + Polymarket contract metadata")
    kalshi = generate_kalshi_contracts(events, latents, consensus, band_probs, cfg, rng)
    poly = generate_polymarket_contracts(events, latents, consensus, band_probs, cfg, rng)
    _write_csv(kalshi, out / "kalshi_contracts.csv")
    _write_csv(poly, out / "polymarket_contracts.csv")

    print("[5/11] contract_event_map")
    cmap = build_contract_event_map(events, kalshi, poly)
    _write_csv(cmap, out / "contract_event_map.csv")

    print("[6/11] Prediction-market 1-min OHLCV (event windows)")
    kalshi_ohlcv = generate_kalshi_ohlcv(kalshi, events, latents, band_probs, cfg, rng)
    poly_ohlcv = generate_polymarket_ohlcv(poly, events, latents, band_probs, cfg, rng)
    _write_csv(kalshi_ohlcv, out / "kalshi_1min_ohlcv.csv")
    _write_csv(poly_ohlcv, out / "polymarket_1min_ohlcv.csv")

    print("[7/11] Daily equity prices + VIX")
    daily_eq = generate_daily_equity_prices(events, latents, cfg, rng)
    vix = generate_vix_daily(events, latents, cfg, rng)
    _write_csv(daily_eq, out / "daily_equity_prices.csv")
    _write_csv(vix, out / "vix_daily.csv")

    print("[8/11] Event signals (pre-release; no returns dependency)")
    signals = generate_event_signals(
        events,
        latents,
        consensus,
        band_probs,
        kalshi_ohlcv,
        poly_ohlcv,
        kalshi,
        poly,
        vix,
        daily_eq,
        cfg,
        rng,
    )
    _write_csv(signals, out / "event_signals.csv")

    print("[9/11] Event-window returns from signals")
    returns = generate_event_returns(events, latents, signals, cfg, rng)
    _write_csv(returns, out / "event_returns.csv")

    print("[10/11] Equity 1-min OHLCV bridged to returns")
    eq_1min, returns = generate_equity_1min_ohlcv(events, daily_eq, returns, cfg, rng)
    _write_csv(eq_1min, out / "equity_1min_ohlcv.csv")
    _write_csv(returns, out / "event_returns.csv")

    print("[11/11] Validation + manifest")
    messages = run_validations(
        events=events,
        latents=latents,
        band_probs=band_probs,
        signals=signals,
        event_returns=returns,
        equity_1min=eq_1min,
        daily_equity=daily_eq,
        contract_map=cmap,
        kalshi=kalshi,
        polymarket=poly,
        consensus=consensus,
        cfg=cfg,
    )
    for msg in messages:
        print(f"  {msg}")

    frames = {
        "events.csv": events,
        "events_latent.csv": latents,
        "consensus_forecasts.csv": consensus,
        "event_band_probabilities.csv": band_probs,
        "kalshi_contracts.csv": kalshi,
        "polymarket_contracts.csv": poly,
        "contract_event_map.csv": cmap,
        "kalshi_1min_ohlcv.csv": kalshi_ohlcv,
        "polymarket_1min_ohlcv.csv": poly_ohlcv,
        "event_signals.csv": signals,
        "daily_equity_prices.csv": daily_eq,
        "equity_1min_ohlcv.csv": eq_1min,
        "event_returns.csv": returns,
        "vix_daily.csv": vix,
    }
    manifest_path = write_manifest(out, cfg, frames, messages)
    print(f"  wrote {manifest_path.name}")
    print(f"Done in {time.time() - t0:.1f}s -> {out}")
    return frames


def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG.with_overrides(
        seed=args.seed,
        start_date=args.start,
        end_date=args.end,
        out_dir=args.out_dir,
    )
    generate_all(cfg)


if __name__ == "__main__":
    main()
