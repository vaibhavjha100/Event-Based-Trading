"""Save model pickles and training summary."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def save_model_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_artifact(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def write_training_outputs(
    *,
    models_dir: Path,
    results: Dict[str, Dict[str, Any]],
    feature_sets: Dict[str, List[str]],
    target: str,
    source_paths: Dict[str, str],
    order_check: Dict[str, Any],
    n_train_base: int,
    expanding_window_note: str,
) -> Dict[str, Any]:
    """Write pickles, coefficients CSV, and training_summary.json."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    coef_rows = []
    models_summary = {}

    for mid, res in sorted(results.items()):
        models_summary[mid] = {
            "status": res["status"],
            "config_id": res["config_id"],
            "estimator_name": res["estimator_name"],
            "feature_names": res["feature_names"],
            "n_rows_used": res["n_rows_used"],
            "target": res["target"],
            "reason": res.get("reason"),
            "extra": {
                k: v
                for k, v in (res.get("extra") or {}).items()
                if k != "coefficients"
            },
        }
        if res["status"] == "trained" and res.get("artifact"):
            art = res["artifact"]
            save_model_artifact(models_dir / f"{mid}.pkl", art)
            # verify round-trip
            loaded = load_model_artifact(models_dir / f"{mid}.pkl")
            if loaded.get("feature_names") != art["feature_names"]:
                raise RuntimeError(f"pickle round-trip feature_names mismatch for {mid}")

            coefs = (res.get("extra") or {}).get("coefficients") or {}
            for fname, val in coefs.items():
                coef_rows.append(
                    {
                        "model_id": mid,
                        "config_id": res["config_id"],
                        "estimator_name": res["estimator_name"],
                        "feature": fname,
                        "coefficient": val,
                    }
                )

    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(models_dir / "coefficients_summary.csv", index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "n_train_base_rows": n_train_base,
        "source_paths": source_paths,
        "order_check": order_check,
        "imputation": "SimpleImputer(strategy='median') inside each Pipeline; fit on that model's training rows only",
        "expanding_window": expanding_window_note,
        "feature_sets": feature_sets,
        "models": models_summary,
        "test_data_accessed": False,
    }
    (models_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
