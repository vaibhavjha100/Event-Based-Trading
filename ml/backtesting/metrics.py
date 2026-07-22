"""Backtest metrics: RMSE, R2, directional accuracy, PnL, drawdown, Sharpe-like."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ml.backtest_config import EPS, SHARPE_CLIP_SCALE


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of events where sign(pred) matches sign(actual); zeros excluded from denom."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    # Require non-zero actual for directional call; pred zero counts as miss if actual nonzero
    usable = yt != 0
    if usable.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(yp[usable]) == np.sign(yt[usable])))


def sharpe_like(pnl: np.ndarray) -> float:
    """mean(pnl) / (std(pnl)+eps) using finite trade or all-event PnL."""
    x = np.asarray(pnl, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.mean(x) / (np.std(x, ddof=0) + EPS))


def sharpe_like_clipped(pnl: np.ndarray, scale: float = SHARPE_CLIP_SCALE) -> float:
    s = sharpe_like(pnl)
    if not np.isfinite(s):
        return 0.0
    return float(np.clip(s / scale, 0.0, 1.0))


def max_drawdown(cum: np.ndarray) -> float:
    """Max peak-to-trough drawdown on cumulative PnL path (absolute units)."""
    x = np.asarray(cum, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    peak = np.maximum.accumulate(x)
    dd = peak - x
    return float(np.max(dd))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signals: np.ndarray,
    pnl: np.ndarray,
) -> Dict[str, Any]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    trade_mask = np.array([s in ("Long", "Short") for s in signals]) & np.isfinite(pnl)
    n_trades = int(trade_mask.sum())
    trade_pnl = pnl[trade_mask]

    cum = np.nancumsum(np.where(np.isfinite(pnl), pnl, 0.0))
    cum_ret = float(cum[-1]) if len(cum) else float("nan")

    r2_raw = r2_score(y_true, y_pred)
    return {
        "n_events": int(m.sum()),
        "n_trades": n_trades,
        "n_long": int(((signals == "Long") & np.isfinite(pnl)).sum()),
        "n_short": int(((signals == "Short") & np.isfinite(pnl)).sum()),
        "n_neutral": int(((signals == "Neutral") & np.isfinite(pnl)).sum()),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_raw,
        "r2_clipped": float(np.clip(r2_raw, 0.0, 1.0)) if np.isfinite(r2_raw) else 0.0,
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "cumulative_return": cum_ret,
        "mean_pnl": float(np.nanmean(pnl)) if np.isfinite(pnl).any() else float("nan"),
        "mean_trade_pnl": float(np.mean(trade_pnl)) if n_trades else float("nan"),
        "sharpe_like": sharpe_like(trade_pnl if n_trades else pnl),
        "sharpe_like_clipped": sharpe_like_clipped(trade_pnl if n_trades else pnl),
        "max_drawdown": max_drawdown(cum),
        "hit_rate_trades": (
            float(np.mean(trade_pnl > 0)) if n_trades else float("nan")
        ),
    }
