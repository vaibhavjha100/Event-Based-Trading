"""Train-only thresholds -> Long / Neutral / Short signals and event PnL."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ml.backtest_config import TAU_FLOOR, TAU_STD_FRAC


def fit_threshold(yhat_train: np.ndarray) -> float:
    """τ = max(1e-6, 0.5 * std(ŷ_train)) from finite train predictions."""
    arr = np.asarray(yhat_train, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return TAU_FLOOR
    return float(max(TAU_FLOOR, TAU_STD_FRAC * float(np.std(arr, ddof=0))))


def apply_signals(yhat: np.ndarray, tau: float) -> np.ndarray:
    """Long if ŷ >= +τ; Short if ŷ <= -τ; else Neutral."""
    out = np.full(len(yhat), "Neutral", dtype=object)
    finite = np.isfinite(yhat)
    out[finite & (yhat >= tau)] = "Long"
    out[finite & (yhat <= -tau)] = "Short"
    return out


def event_pnl(signals: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Long → +R, Short → −R, Neutral → 0. No transaction costs."""
    R = np.asarray(realized, dtype=float)
    pnl = np.zeros(len(signals), dtype=float)
    for i, s in enumerate(signals):
        if not np.isfinite(R[i]):
            pnl[i] = np.nan
            continue
        if s == "Long":
            pnl[i] = R[i]
        elif s == "Short":
            pnl[i] = -R[i]
        else:
            pnl[i] = 0.0
    return pnl


def signal_and_pnl(
    yhat: np.ndarray, realized: np.ndarray, tau: float
) -> Tuple[np.ndarray, np.ndarray]:
    signals = apply_signals(yhat, tau)
    pnl = event_pnl(signals, realized)
    return signals, pnl
