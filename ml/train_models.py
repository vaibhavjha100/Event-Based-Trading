"""Train M0–M3 models on the training partition only.

Usage
-----
python -m ml.train_models
python -m ml.train_models --target ret_SPY_60m --models-dir models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.train_config import DEFAULT_TRAIN_CONFIG
from ml.training.feature_sets import build_feature_sets
from ml.training.fit import fit_all_models
from ml.training.io_utils import write_training_outputs
from ml.training.load_ml_data import load_train_bundle
from ml.training.validate_training import validate_training_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M0–M3 models on training data only.")
    p.add_argument("--ml-dir", type=str, default=None)
    p.add_argument("--cleaned-dir", type=str, default=None)
    p.add_argument("--models-dir", type=str, default=None)
    p.add_argument("--target", type=str, default=None)
    return p.parse_args()


def train_models(cfg=None) -> dict:
    """Run the full train-only fitting pipeline."""
    cfg = cfg or DEFAULT_TRAIN_CONFIG

    print("[1] Load training data (no test files)")
    bundle = load_train_bundle(cfg)
    print(f"  n_rows={len(bundle.frame)} target={bundle.target}")
    print(f"  order_check={bundle.order_check}")

    print("[2] Build M0-M3 feature sets")
    feature_sets = build_feature_sets(bundle.feature_columns_available)
    for k, v in feature_sets.items():
        print(f"  {k}: {len(v)} features")

    print("[3] Fit models (OLS, ElasticNet, GBDT) per config")
    results = fit_all_models(bundle.frame, feature_sets, bundle.target, cfg)
    for mid, res in sorted(results.items()):
        status = res["status"]
        n = res["n_rows_used"]
        extra = f" reason={res.get('reason')}" if status != "trained" else ""
        print(f"  {mid}: {status} n_rows={n}{extra}")

    print(f"[4] Write artifacts -> {cfg.models_dir}")
    expanding_note = (
        f"{cfg.n_expanding_folds} chronological expanding-window folds on "
        "training rows only (no shuffle); used for ElasticNet and GBDT HP selection"
    )
    summary = write_training_outputs(
        models_dir=Path(cfg.models_dir),
        results=results,
        feature_sets=feature_sets,
        target=bundle.target,
        source_paths=bundle.source_paths,
        order_check=bundle.order_check,
        n_train_base=len(bundle.frame),
        expanding_window_note=expanding_note,
    )

    print("[5] Validate")
    messages = validate_training_run(
        feature_sets=feature_sets,
        results=results,
        models_dir=Path(cfg.models_dir),
        n_train_base=len(bundle.frame),
    )
    for msg in messages:
        print(f"  {msg}")

    print("Done. test_data_accessed=False")
    return summary


def main() -> None:
    args = parse_args()
    kwargs = {}
    if args.ml_dir:
        kwargs["ml_dir"] = Path(args.ml_dir)
    if args.cleaned_dir:
        kwargs["cleaned_dir"] = Path(args.cleaned_dir)
    if args.models_dir:
        kwargs["models_dir"] = Path(args.models_dir)
    if args.target:
        kwargs["target"] = args.target
    cfg = DEFAULT_TRAIN_CONFIG.with_overrides(**kwargs)
    train_models(cfg)


if __name__ == "__main__":
    main()
