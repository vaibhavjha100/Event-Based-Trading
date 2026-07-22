"""Shared IO helpers for backtest outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def package_versions(keys: tuple) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in keys:
        try:
            if key == "scikit-learn":
                import sklearn

                out[key] = sklearn.__version__
            elif key == "numpy":
                import numpy as np

                out[key] = np.__version__
            elif key == "pandas":
                out[key] = pd.__version__
            elif key == "lightgbm":
                import lightgbm as lgb

                out[key] = lgb.__version__
            else:
                out[key] = "unknown"
        except Exception:
            out[key] = "not_installed"
    return out
