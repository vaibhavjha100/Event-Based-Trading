"""Explicit multi-metric model selection rule."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.backtest_config import MIN_TRADES, W_DIR_ACC, W_R2, W_SHARPE

SELECTION_RULE_TEXT = (
    f"Among models with n_trades >= {MIN_TRADES}: "
    f"score = {W_DIR_ACC:.2f} * directional_accuracy + "
    f"{W_SHARPE:.2f} * sharpe_like_clipped + {W_R2:.2f} * r2_clipped; "
    "if none qualify, drop trade filter. "
    "Tie-break: higher cumulative_return, then lower RMSE. "
    "Thresholds from train-only predictions; walkforward_mode=expanding_refit_fixed_hparams."
)


def selection_score(metrics: Dict[str, Any]) -> float:
    da = metrics.get("directional_accuracy")
    sh = metrics.get("sharpe_like_clipped", 0.0)
    r2c = metrics.get("r2_clipped", 0.0)
    if da is None or not np.isfinite(da):
        da = 0.0
    if not np.isfinite(sh):
        sh = 0.0
    if not np.isfinite(r2c):
        r2c = 0.0
    return float(W_DIR_ACC * da + W_SHARPE * sh + W_R2 * r2c)


def select_best_model(
    metrics_by_model: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return (best_model_id, selection_info)."""
    if not metrics_by_model:
        return None, {"reason": "no models scored", "rule": SELECTION_RULE_TEXT}

    scored: List[Tuple[str, float, Dict[str, Any]]] = []
    for mid, m in metrics_by_model.items():
        sc = selection_score(m)
        m = dict(m)
        m["selection_score"] = sc
        scored.append((mid, sc, m))

    eligible = [(mid, sc, m) for mid, sc, m in scored if m.get("n_trades", 0) >= MIN_TRADES]
    used_filter = True
    pool = eligible if eligible else scored
    if not eligible:
        used_filter = False

    def sort_key(item: Tuple[str, float, Dict[str, Any]]):
        mid, sc, m = item
        cum = m.get("cumulative_return", float("-inf"))
        if not np.isfinite(cum):
            cum = float("-inf")
        rmse = m.get("rmse", float("inf"))
        if not np.isfinite(rmse):
            rmse = float("inf")
        # higher score, higher cum, lower rmse
        return (sc, cum, -rmse)

    pool_sorted = sorted(pool, key=sort_key, reverse=True)
    best_id, best_sc, best_m = pool_sorted[0]

    info = {
        "selected_model_id": best_id,
        "selection_score": best_sc,
        "n_trades_filter_applied": used_filter,
        "min_trades": MIN_TRADES,
        "n_candidates": len(scored),
        "n_eligible": len(eligible) if used_filter else len(scored),
        "rule": SELECTION_RULE_TEXT,
        "selection_reason": (
            f"Selected {best_id} with selection_score={best_sc:.4f} "
            f"(dir_acc={best_m.get('directional_accuracy')}, "
            f"sharpe_clipped={best_m.get('sharpe_like_clipped')}, "
            f"r2_clipped={best_m.get('r2_clipped')}, "
            f"n_trades={best_m.get('n_trades')}, "
            f"cum_ret={best_m.get('cumulative_return')}, "
            f"rmse={best_m.get('rmse')}). "
            f"Trade filter {'applied' if used_filter else 'waived (no model had >= ' + str(MIN_TRADES) + ' trades)'}. "
            + SELECTION_RULE_TEXT
        ),
    }
    # write scores back
    for mid, sc, m in scored:
        metrics_by_model[mid]["selection_score"] = sc

    return best_id, info
