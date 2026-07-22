"""Training-time validation checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ml.training.feature_sets import assert_no_leakage


def validate_training_run(
    *,
    feature_sets: Dict[str, List[str]],
    results: Dict[str, Dict[str, Any]],
    models_dir: Path,
    n_train_base: int,
) -> List[str]:
    """Return list of validation messages; raise on hard failures."""
    messages: List[str] = []
    errors: List[str] = []

    if n_train_base <= 0:
        errors.append("n_train_base must be > 0")

    # Nested feature counts should be non-decreasing when columns present
    sizes = {k: len(v) for k, v in feature_sets.items()}
    messages.append(f"feature_counts={sizes}")
    if sizes.get("M0", 0) < 1:
        errors.append("M0 has no features")
    if sizes.get("M1", 0) < sizes.get("M0", 0):
        errors.append("M1 feature count < M0")
    if sizes.get("M2", 0) < sizes.get("M1", 0):
        errors.append("M2 feature count < M1")
    if sizes.get("M3", 0) < sizes.get("M2", 0):
        errors.append("M3 feature count < M2")

    for cfg_id, feats in feature_sets.items():
        try:
            assert_no_leakage(feats)
            messages.append(f"{cfg_id}: leakage check passed ({len(feats)} features)")
        except ValueError as exc:
            errors.append(str(exc))

    trained = 0
    for mid, res in results.items():
        if res["status"] == "trained":
            trained += 1
            path = models_dir / f"{mid}.pkl"
            if not path.exists():
                errors.append(f"missing pickle for {mid}")
            art = res.get("artifact") or {}
            if art.get("feature_names") != res.get("feature_names"):
                errors.append(f"feature_names mismatch in artifact for {mid}")
            if not art.get("feature_names"):
                errors.append(f"empty feature_names for {mid}")
            if res.get("n_rows_used", 0) <= 0:
                errors.append(f"n_rows_used <= 0 for {mid}")
        else:
            messages.append(f"skipped {mid}: {res.get('reason')}")

    messages.append(f"trained_models={trained}/{len(results)}")
    if trained == 0:
        errors.append("no models trained successfully")

    if errors:
        raise RuntimeError("Training validation failed: " + "; ".join(errors))
    messages.append("All training validation checks passed.")
    return messages
