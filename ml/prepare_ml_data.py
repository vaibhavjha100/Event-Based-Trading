"""Prepare ML-ready feature/target tables with chronological train/test split.

Usage
-----
python -m ml.prepare_ml_data
python -m ml.prepare_ml_data --train-ratio 0.75 --cleaned-dir data/cleaned --out-dir data/ml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.ml_data_config import DEFAULT_ML_DATA_CONFIG
from ml.ml_dataset.features import (
    build_feature_matrix,
    feature_columns_table,
    filter_rows_for_core_and_targets,
    resolve_feature_columns,
)
from ml.ml_dataset.load import load_and_prepare_events
from ml.ml_dataset.split import chronological_split
from ml.ml_dataset.targets import resolve_target_columns
from ml.ml_dataset.write import write_ml_artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build chronological train/test ML datasets from cleaned event-level data."
    )
    p.add_argument("--cleaned-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument(
        "--include-additional-targets",
        action="store_true",
        help="Also export additional ret_* targets beyond SPY/QQQ 60m.",
    )
    return p.parse_args()


def prepare_ml_data(cfg=None) -> dict:
    """Run the full ML dataset preparation pipeline."""
    cfg = cfg or DEFAULT_ML_DATA_CONFIG

    print(f"[1] Load event-level table: {cfg.event_level_path}")
    df = load_and_prepare_events(cfg)
    dropped_noise = int(df.attrs.get("dropped_noise_rows", 0))
    source_path = str(df.attrs.get("source_path", cfg.event_level_path))
    print(f"  rows after dropping NOISE: {len(df)} (dropped {dropped_noise})")

    print("[2] Resolve feature and target columns")
    core_used, optional_used, feature_meta = resolve_feature_columns(df, cfg)
    target_columns = resolve_target_columns(df, cfg)
    feature_columns = feature_meta["feature_columns_ordered"]
    print(f"  core features: {len(core_used)}")
    print(f"  optional features present: {len(optional_used)}")
    print(f"  targets: {target_columns}")

    print("[3] Filter rows (non-null primary targets + core features)")
    n_before = len(df)
    df = filter_rows_for_core_and_targets(df, core_used, cfg.primary_targets)
    print(f"  rows retained: {len(df)} / {n_before}")

    print(f"[4] Chronological split train_ratio={cfg.train_ratio}")
    split = chronological_split(df, train_ratio=cfg.train_ratio)
    print(
        f"  train={split.n_train} test={split.n_test} "
        f"boundary={split.split_boundary_release_time}"
    )

    # Ensure matrices exist (side-effect free validation)
    _ = build_feature_matrix(split.train, feature_columns)
    _ = build_feature_matrix(split.test, feature_columns)

    print(f"[5] Write artifacts -> {cfg.out_dir}")
    universe_rule = (
        f"exclude event_type in {list(cfg.exclude_event_types)}; "
        "require non-null primary targets and core features"
    )
    meta = write_ml_artifacts(
        out_dir=Path(cfg.out_dir),
        split=split,
        feature_columns=feature_columns,
        feature_groups=feature_columns_table(core_used, optional_used),
        target_columns=target_columns,
        feature_meta=feature_meta,
        source_path=source_path,
        universe_rule=universe_rule,
        dropped_noise_rows=dropped_noise,
        n_before_core_filter=n_before,
    )

    print("Done.")
    print(f"  X_train: {split.n_train} x {len(feature_columns)}")
    print(f"  X_test:  {split.n_test} x {len(feature_columns)}")
    print(f"  y columns: {target_columns}")
    return meta


def main() -> None:
    args = parse_args()
    kwargs = {}
    if args.cleaned_dir:
        kwargs["cleaned_dir"] = Path(args.cleaned_dir)
    if args.out_dir:
        kwargs["out_dir"] = Path(args.out_dir)
    if args.train_ratio is not None:
        kwargs["train_ratio"] = args.train_ratio
    if args.include_additional_targets:
        kwargs["include_additional_targets"] = True
    cfg = DEFAULT_ML_DATA_CONFIG.with_overrides(**kwargs)
    prepare_ml_data(cfg)


if __name__ == "__main__":
    main()
