from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference import infer_live_signals


REQUIRED_SIGNAL_COLUMNS = {
    "date",
    "market_title",
    "category",
    "target_asset",
    "p_bad",
    "pmps_pre",
    "volume",
    "open_interest",
    "signal_direction",
    "model_forecast",
    "classification_confidence",
    "known_at",
}
REQUIRED_PERFORMANCE_COLUMNS = {
    "date",
    "strategy",
    "benchmark",
    "drawdown",
    "strategy_return",
    "benchmark_return",
}
REQUIRED_METRIC_KEYS = {
    "model_version",
    "data_updated_at",
    "source",
    "train_period",
    "validation_period",
    "test_period",
    "strategy_return",
    "benchmark_return",
    "sharpe",
    "max_drawdown",
    "hit_rate",
    "number_of_trades",
    "transaction_cost_bps",
    "test_r2",
    "test_directional_accuracy",
}


def _read_csv(data_dir: Path, name: str) -> pd.DataFrame | None:
    path = data_dir / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json(data_dir: Path, name: str) -> dict | None:
    path = data_dir / name
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def demo_signals() -> pd.DataFrame:
    dates = pd.date_range("2024-10-01", "2024-12-31", freq="D")
    rows: list[dict] = []
    specs = [
        ("CPI", "Will US CPI exceed 3.0%?", "SPY", 0.44, 0.07),
        ("Fed", "Will the Fed keep rates above 5%?", "SPY", 0.56, -0.05),
    ]
    for i, (category, title, asset, base, drift) in enumerate(specs):
        phase = np.linspace(0, 3 * np.pi, len(dates)) + i
        p_bad = np.clip(
            base + 0.08 * np.sin(phase) + drift * np.linspace(0, 1, len(dates)),
            0.05,
            0.95,
        )
        volume = 3500 + (i + 1) * 900 + 1800 * (1 + np.sin(phase + 0.7))
        open_interest = 1500 + (i + 1) * 500 + 700 * (1 + np.cos(phase))
        for index, date in enumerate(dates):
            pmps_pre = 0.0 if index == 0 else p_bad[index] - p_bad[index - 1]
            direction = (
                "Short"
                if pmps_pre > 0.012
                else "Long"
                if pmps_pre < -0.012
                else "Neutral"
            )
            rows.append(
                {
                    "date": date,
                    "market_title": title,
                    "category": category,
                    "target_asset": asset,
                    "p_bad": p_bad[index],
                    "pmps_pre": pmps_pre,
                    "volume": round(volume[index]),
                    "open_interest": round(open_interest[index]),
                    "signal_direction": direction,
                    "model_forecast": -0.045 * pmps_pre
                    + (0.0003 if asset == "SPY" else 0.0001),
                    "classification_confidence": 0.94 - i * 0.03,
                    "known_at": f"{date.date()} 08:25:00 America/New_York",
                }
            )
    return pd.DataFrame(rows)


def demo_performance() -> pd.DataFrame:
    dates = pd.bdate_range("2023-07-03", "2024-12-31")
    rng = np.random.default_rng(5557)
    strategy_return = rng.normal(0.00042, 0.0065, len(dates))
    benchmark_return = rng.normal(0.00032, 0.0078, len(dates))
    strategy = 100 * np.cumprod(1 + strategy_return)
    benchmark = 100 * np.cumprod(1 + benchmark_return)
    peak = np.maximum.accumulate(strategy)
    return pd.DataFrame(
        {
            "date": dates,
            "strategy": strategy,
            "benchmark": benchmark,
            "drawdown": strategy / peak - 1,
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
        }
    )


def demo_metrics() -> dict:
    return {
        "model_version": "demo-v0.2",
        "data_updated_at": "2024-12-31 08:25 America/New_York",
        "source": "DEMO DATA - awaiting validated team handoff",
        "train_period": "2020-01-01 to 2022-12-31",
        "validation_period": "2023-01-01 to 2023-06-30",
        "test_period": "2023-07-01 to 2024-12-31",
        "strategy_return": 0.126,
        "benchmark_return": 0.091,
        "sharpe": 1.18,
        "max_drawdown": -0.072,
        "hit_rate": 0.56,
        "number_of_trades": 84,
        "transaction_cost_bps": 10,
        "test_r2": 0.041,
        "test_directional_accuracy": 0.554,
    }


def _selected_model_row(data_dir: Path) -> tuple[pd.Series | None, pd.DataFrame | None]:
    comparison = _read_csv(data_dir, "model_comparison.csv")
    if comparison is None or comparison.empty:
        return None, comparison
    metadata = _read_json(data_dir, "model_metadata.json") or {}
    model_id = metadata.get("model_id")
    if model_id and "model_id" in comparison.columns:
        match = comparison.loc[comparison["model_id"] == model_id]
        if not match.empty:
            return match.iloc[0], comparison
    if "selected" in comparison.columns:
        selected = comparison.loc[comparison["selected"] == True]
        if not selected.empty:
            return selected.iloc[0], comparison
    return comparison.iloc[0], comparison


def _baseline_row(comparison: pd.DataFrame | None) -> pd.Series | None:
    if comparison is None or comparison.empty:
        return None
    if "model_id" in comparison.columns:
        preferred = comparison.loc[comparison["model_id"] == "M0_ols"]
        if not preferred.empty:
            return preferred.iloc[0]
    if "config_id" in comparison.columns:
        baseline = comparison.loc[comparison["config_id"] == "M0"]
        if not baseline.empty:
            return baseline.iloc[0]
    return None


def _period_text(value: object) -> str:
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(f"{key}: {item}")
        return "; ".join(parts)
    return str(value) if value is not None else "Not supplied"


def _transaction_cost_bps(metadata: dict) -> int:
    text = str(metadata.get("methodology_config", {}).get("transaction_costs", ""))
    if "none" in text.lower():
        return 0
    return 0


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def adapt_model_metrics(data_dir: Path) -> dict | None:
    row, comparison = _selected_model_row(data_dir)
    if row is None:
        return None

    metadata = _read_json(data_dir, "model_metadata.json") or {}
    baseline = _baseline_row(comparison)
    data_quality = _read_csv(data_dir, "data_quality_summary.csv")
    quality_map = {}
    if data_quality is not None and {"item", "value"}.issubset(data_quality.columns):
        quality_map = dict(zip(data_quality["item"], data_quality["value"]))

    benchmark_return = (
        float(baseline["cumulative_return"])
        if baseline is not None and "cumulative_return" in baseline
        else 0.0
    )
    test_period = metadata.get("test_period", {})
    test_label = quality_map.get("test_period_label", "historical_walkforward_test")
    test_period_text = _period_text(test_period)
    if quality_map.get("split_boundary_release_time"):
        test_period_text = (
            f"{test_label}; {test_period_text}; "
            f"split: {quality_map['split_boundary_release_time']}"
        )

    return {
        "model_version": metadata.get("model_id")
        or metadata.get("version")
        or str(row.get("model_id", "Not supplied")),
        "data_updated_at": metadata.get("generated_at", "Not supplied"),
        "source": "Data handoff output: simulated scenario signals plus historical walk-forward backtest",
        "train_period": _period_text(metadata.get("train_period")),
        "validation_period": _period_text(metadata.get("val_period")),
        "test_period": test_period_text,
        "strategy_return": float(row.get("cumulative_return", 0.0)),
        "benchmark_return": benchmark_return,
        "sharpe": float(row.get("sharpe_like", 0.0)),
        "max_drawdown": -abs(float(row.get("max_drawdown", 0.0))),
        "hit_rate": float(row.get("directional_accuracy", 0.0)),
        "number_of_trades": int(row.get("n_trades", 0)),
        "transaction_cost_bps": _transaction_cost_bps(metadata),
        "transaction_cost_note": quality_map.get("transaction_costs")
        or metadata.get("methodology_config", {}).get("transaction_costs", ""),
        "test_r2": float(row.get("r2", 0.0)),
        "test_rmse": float(row.get("rmse", 0.0)),
        "test_directional_accuracy": float(row.get("directional_accuracy", 0.0)),
        "selected_model_id": str(row.get("model_id", "")),
        "data_selected_model_id": quality_map.get("selected_model_id", ""),
        "target_variable": metadata.get("target")
        or quality_map.get("primary_target", "Not supplied"),
        "walkforward_mode": metadata.get("walkforward_mode")
        or quality_map.get("walkforward_mode", "Not supplied"),
        "n_train_events": _safe_int(quality_map.get("n_train")),
        "n_test_events": _safe_int(quality_map.get("n_test")),
        "n_model_artifacts": _safe_int(quality_map.get("n_artifacts_scored")),
        "selection_reason": metadata.get("selection_reason", ""),
        "selection_rule": metadata.get("selection_rule", ""),
        "surprise_timing": metadata.get("surprise_timing", ""),
        "date_range_note": metadata.get("date_range_note", ""),
        "simulated_rows_label": quality_map.get("simulated_rows_label", ""),
        "entry_exit": quality_map.get("entry_exit")
        or metadata.get("methodology_config", {}).get("entry_exit", ""),
    }


def adapt_cumulative_returns(
    performance: pd.DataFrame | None, data_dir: Path
) -> pd.DataFrame | None:
    if performance is None:
        return None
    if not {"release_time", "selected"}.issubset(performance.columns):
        return None

    frame = performance.copy()
    frame["date"] = pd.to_datetime(frame["release_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if frame.empty:
        return None

    benchmark_col = "M0_ols" if "M0_ols" in frame.columns else None
    if benchmark_col is None:
        m0_cols = [column for column in frame.columns if column.startswith("M0_")]
        benchmark_col = m0_cols[0] if m0_cols else "selected"

    metadata = _read_json(data_dir, "model_metadata.json") or {}
    strategy_col = metadata.get("model_id") if metadata.get("model_id") in frame.columns else "selected"
    strategy_cum = pd.to_numeric(frame[strategy_col], errors="coerce").fillna(0.0)
    benchmark_cum = pd.to_numeric(frame[benchmark_col], errors="coerce").fillna(0.0)
    strategy = 100 * (1 + strategy_cum)
    benchmark = 100 * (1 + benchmark_cum)
    drawdown = strategy / strategy.cummax() - 1

    return pd.DataFrame(
        {
            "date": frame["date"],
            "strategy": strategy,
            "benchmark": benchmark,
            "drawdown": drawdown,
            "strategy_return": strategy_cum.diff().fillna(strategy_cum),
            "benchmark_return": benchmark_cum.diff().fillna(benchmark_cum),
        }
    )


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, bool, list[str]]:
    signals = _read_csv(data_dir, "live_signals.csv")
    performance = _read_csv(data_dir, "cumulative_returns.csv")
    metrics = _read_json(data_dir, "model_metrics.json")
    issues: list[str] = []

    signal_columns = set() if signals is None else set(signals.columns)
    performance_columns = set() if performance is None else set(performance.columns)
    metric_keys = set() if metrics is None else set(metrics)

    missing_signal_columns = REQUIRED_SIGNAL_COLUMNS - signal_columns
    missing_performance_columns = REQUIRED_PERFORMANCE_COLUMNS - performance_columns
    missing_metric_keys = REQUIRED_METRIC_KEYS - metric_keys

    signal_issue = None
    if signals is None:
        signal_issue = "live_signals.csv is missing"
    elif missing_signal_columns:
        signal_issue = (
            f"live_signals.csv is missing: {', '.join(sorted(missing_signal_columns))}"
        )

    if signal_issue:
        inferred_signals, inference_issues = infer_live_signals(data_dir)
        if inferred_signals is not None:
            signals = inferred_signals
            missing_signal_columns = REQUIRED_SIGNAL_COLUMNS - set(signals.columns)
            signal_issue = None if not missing_signal_columns else (
                "inferred live signals are missing: "
                + ", ".join(sorted(missing_signal_columns))
            )
        if signal_issue:
            issues.append(signal_issue)
            issues.extend(inference_issues)
    performance_issue = None
    if performance is None:
        performance_issue = "cumulative_returns.csv is missing"
    elif missing_performance_columns:
        performance_issue = (
            "cumulative_returns.csv is missing: "
            + ", ".join(sorted(missing_performance_columns))
        )
    if performance_issue:
        adapted_performance = adapt_cumulative_returns(performance, data_dir)
        if adapted_performance is not None:
            performance = adapted_performance
            missing_performance_columns = REQUIRED_PERFORMANCE_COLUMNS - set(
                performance.columns
            )
            performance_issue = None if not missing_performance_columns else (
                "adapted cumulative_returns.csv is missing: "
                + ", ".join(sorted(missing_performance_columns))
            )
        if performance_issue:
            issues.append(performance_issue)

    metrics_issue = None
    if metrics is None:
        metrics_issue = "model_metrics.json is missing"
    elif missing_metric_keys:
        metrics_issue = (
            f"model_metrics.json is missing: {', '.join(sorted(missing_metric_keys))}"
        )
    if metrics_issue:
        adapted_metrics = adapt_model_metrics(data_dir)
        if adapted_metrics is not None:
            metrics = adapted_metrics
            missing_metric_keys = REQUIRED_METRIC_KEYS - set(metrics)
            metrics_issue = None if not missing_metric_keys else (
                "adapted model metrics are missing: "
                + ", ".join(sorted(missing_metric_keys))
            )
        if metrics_issue:
            issues.append(metrics_issue)

    if signals is None or missing_signal_columns:
        signals = demo_signals()
    if performance is None or missing_performance_columns:
        performance = demo_performance()
    if metrics is None or missing_metric_keys:
        metrics = demo_metrics()

    signals["date"] = pd.to_datetime(signals["date"])
    performance["date"] = pd.to_datetime(performance["date"])
    return signals, performance, metrics, bool(issues), issues
