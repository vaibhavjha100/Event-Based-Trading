"""CLI: walk-forward score all models, select best, export research + website artifacts.

Usage:
    python -m ml.run_backtest
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml.backtest_config import (
    OUTPUT_DIR,
    PRIMARY_TARGET,
    WALKFORWARD_MODE,
)
from ml.backtesting.io_utils import ensure_dir, write_csv, write_json
from ml.backtesting.load_artifacts import load_artifacts, load_split_frames
from ml.backtesting.metrics import compute_metrics
from ml.backtesting.prepare_test_data import build_model_matrices
from ml.backtesting.select_best_model import select_best_model
from ml.backtesting.signal_policy import fit_threshold, signal_and_pnl
from ml.backtesting.walkforward import fit_full_train_predictions, walkforward_expanding_refit
from ml.backtesting.website_bundle import SklearnReturnModel, write_website_outputs


def _score_one_model(data, artifact: Dict[str, Any]) -> Dict[str, Any]:
    model_id = artifact["model_id"]
    Xtr, Xte, ytr, yte = build_model_matrices(data, artifact)

    yhat_train = fit_full_train_predictions(Xtr, ytr, artifact)
    tau = fit_threshold(yhat_train)

    yhat_test = walkforward_expanding_refit(Xtr, ytr, Xte, yte, artifact)
    signals, pnl = signal_and_pnl(yhat_test, yte.to_numpy(dtype=float), tau)
    metrics = compute_metrics(
        yte.to_numpy(dtype=float), yhat_test, signals, pnl
    )
    metrics["model_id"] = model_id
    metrics["config_id"] = artifact.get("config_id")
    metrics["estimator_name"] = artifact.get("estimator_name")
    metrics["tau"] = tau
    metrics["n_features"] = len(artifact["feature_names"])
    metrics["walkforward_mode"] = WALKFORWARD_MODE

    event_df = pd.DataFrame(
        {
            "event_id": data.test_event_ids.values,
            "release_time": pd.to_datetime(data.test_release_time.values, utc=True),
            "model_id": model_id,
            "y_true": yte.values,
            "y_pred": yhat_test,
            "signal": signals,
            "pnl": pnl,
            "tau": tau,
        }
    )
    return {"metrics": metrics, "events": event_df, "tau": tau, "X_train": Xtr}


def run_backtest(output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    ensure_dir(output_dir)
    data = load_split_frames()
    artifacts, skips = load_artifacts()

    all_events: List[pd.DataFrame] = []
    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    tau_by_model: Dict[str, float] = {}
    X_train_by_model: Dict[str, pd.DataFrame] = {}
    cum_cols: Dict[str, np.ndarray] = {}

    print(f"Loaded {len(artifacts)} artifacts; skipped {len(skips)}")
    for model_id, art in sorted(artifacts.items()):
        print(f"  Scoring {model_id} ({WALKFORWARD_MODE})...")
        result = _score_one_model(data, art)
        metrics_by_model[model_id] = result["metrics"]
        tau_by_model[model_id] = result["tau"]
        X_train_by_model[model_id] = result["X_train"]
        ev = result["events"]
        all_events.append(ev)
        pnl = ev["pnl"].to_numpy(dtype=float)
        cum_cols[model_id] = np.nancumsum(np.where(np.isfinite(pnl), pnl, 0.0))

    best_id, selection_info = select_best_model(metrics_by_model)
    print(f"Selected: {best_id}")
    print(selection_info.get("selection_reason", "")[:200])

    # --- Research outputs ---
    event_results = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    write_csv(event_results, output_dir / "event_results.csv")

    # cumulative returns: date/event order + per model + selected
    events_by_model = {
        str(e["model_id"].iloc[0]): e for e in all_events if len(e)
    }
    if best_id and best_id in events_by_model:
        base_events = events_by_model[best_id]
    elif all_events:
        base_events = all_events[0]
    else:
        base_events = pd.DataFrame()

    cum_df = pd.DataFrame(
        {
            "event_id": base_events["event_id"].values if len(base_events) else [],
            "release_time": base_events["release_time"].values if len(base_events) else [],
        }
    )
    for mid, series in cum_cols.items():
        cum_df[mid] = series
    if best_id and best_id in cum_cols:
        cum_df["selected"] = cum_cols[best_id]
    write_csv(cum_df, output_dir / "cumulative_returns.csv")

    write_json(metrics_by_model, output_dir / "model_metrics.json")

    comparison_rows = []
    for mid, m in metrics_by_model.items():
        comparison_rows.append(
            {
                "model_id": mid,
                "config_id": m.get("config_id"),
                "estimator_name": m.get("estimator_name"),
                "selection_score": m.get("selection_score"),
                "directional_accuracy": m.get("directional_accuracy"),
                "sharpe_like": m.get("sharpe_like"),
                "sharpe_like_clipped": m.get("sharpe_like_clipped"),
                "r2": m.get("r2"),
                "r2_clipped": m.get("r2_clipped"),
                "rmse": m.get("rmse"),
                "cumulative_return": m.get("cumulative_return"),
                "max_drawdown": m.get("max_drawdown"),
                "n_trades": m.get("n_trades"),
                "n_events": m.get("n_events"),
                "tau": m.get("tau"),
                "selected": mid == best_id,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    if len(comparison):
        comparison = comparison.sort_values(
            "selection_score", ascending=False, na_position="last"
        )
    write_csv(comparison, output_dir / "model_comparison.csv")

    # trade log: non-neutral (+ optionally mark neutrals)
    if len(event_results):
        trades = event_results.copy()
        trades["is_trade"] = trades["signal"].isin(["Long", "Short"])
        trade_log = trades[trades["is_trade"]].drop(columns=["is_trade"])
        # also write neutrals for selected model as flagged optional rows? Plan: one row per non-neutral
        write_csv(trade_log, output_dir / "trade_log.csv")
    else:
        write_csv(pd.DataFrame(), output_dir / "trade_log.csv")

    quality = pd.DataFrame(
        [
            {"item": "walkforward_mode", "value": WALKFORWARD_MODE},
            {"item": "primary_target", "value": PRIMARY_TARGET},
            {"item": "n_train", "value": data.split_meta.get("n_train")},
            {"item": "n_test", "value": data.split_meta.get("n_test")},
            {
                "item": "split_boundary_release_time",
                "value": data.split_meta.get("split_boundary_release_time"),
            },
            {"item": "n_artifacts_scored", "value": len(artifacts)},
            {"item": "n_artifacts_skipped", "value": len(skips)},
            {"item": "selected_model_id", "value": best_id},
            {
                "item": "transaction_costs",
                "value": "none (documented assumption)",
            },
            {
                "item": "entry_exit",
                "value": "t0-5 -> t0+60 aligned to ret_SPY_60m",
            },
            {
                "item": "leakage_exclusions",
                "value": "reaction_*, ret_*, actual_value; surprise joined by event_id only",
            },
            {
                "item": "test_period_label",
                "value": "historical_walkforward_test",
            },
            {
                "item": "simulated_rows_label",
                "value": "demo_only_not_historical",
            },
        ]
        + [
            {"item": f"skip_{s['model_id']}", "value": s["reason"]} for s in skips
        ]
    )
    write_csv(quality, output_dir / "data_quality_summary.csv")

    # --- Website handoff ---
    if best_id is None:
        raise RuntimeError("No model selected; cannot export website bundle")

    selected_art = artifacts[best_id]
    # Reload full original pipeline from pickle for website (fitted on train)
    with open(selected_art["path"], "rb") as f:
        selected_full = pickle.load(f)
    selected_full["model_id"] = best_id

    write_website_outputs(
        output_dir=output_dir,
        selected_artifact=selected_full,
        tau=tau_by_model[best_id],
        selection_info=selection_info,
        split_meta=data.split_meta,
        reference_X=X_train_by_model[best_id],
        methodology_config={
            "config_id": selected_full.get("config_id"),
            "estimator_name": selected_full.get("estimator_name"),
            "entry_exit": "event window aligned to target (t0-5 -> t0+60)",
            "transaction_costs": "none",
            "walkforward_mode": WALKFORWARD_MODE,
            "selection_score": selection_info.get("selection_score"),
        },
    )

    # Validate fixture vs bundle
    _validate_outputs(output_dir, best_id)

    return {
        "selected_model_id": best_id,
        "selection_info": selection_info,
        "n_scored": len(artifacts),
        "n_skipped": len(skips),
        "output_dir": str(output_dir),
    }


def _validate_outputs(output_dir: Path, best_id: str) -> None:
    required = [
        "model_bundle.pkl",
        "model_metadata.json",
        "simulated_features.csv",
        "simulation_assumptions.md",
        "prediction_fixture.csv",
        "cumulative_returns.csv",
        "model_metrics.json",
        "event_results.csv",
        "model_comparison.csv",
        "data_quality_summary.csv",
        "trade_log.csv",
    ]
    missing = [f for f in required if not (output_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing outputs: {missing}")

    import json

    with open(output_dir / "model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(output_dir / "model_metadata.json", encoding="utf-8") as f:
        meta_d = json.load(f)

    feats = meta_d["feature_columns"]
    sim = pd.read_csv(output_dir / "simulated_features.csv")
    missing_feats = [c for c in feats if c not in sim.columns]
    if missing_feats:
        raise ValueError(f"simulated_features missing feature_columns: {missing_feats}")

    fixture = pd.read_csv(output_dir / "prediction_fixture.csv")
    model: SklearnReturnModel = bundle["model"]
    X = fixture[feats]
    preds = model.predict(X)
    if not np.allclose(preds, fixture["expected_prediction"].to_numpy(dtype=float), rtol=0, atol=1e-9):
        raise AssertionError("Fixture expected_prediction mismatch vs model_bundle.predict")

    # Leakage audit on selected features
    from ml.backtesting.load_artifacts import _is_leakage

    leak = [c for c in feats if _is_leakage(c)]
    if leak:
        raise AssertionError(f"Leakage columns in selected features: {leak}")

    print(f"Validation OK: 11 outputs present; fixture matches; selected={best_id}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest and model selection")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: output/)",
    )
    args = parser.parse_args(argv)
    summary = run_backtest(output_dir=args.output_dir)
    print(json_summary(summary))


def json_summary(summary: Dict[str, Any]) -> str:
    import json

    return json.dumps(
        {
            "selected_model_id": summary["selected_model_id"],
            "n_scored": summary["n_scored"],
            "n_skipped": summary["n_skipped"],
            "output_dir": summary["output_dir"],
        },
        indent=2,
    )


if __name__ == "__main__":
    main()
