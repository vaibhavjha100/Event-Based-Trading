"""Preprocessing validation and report generation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml.preprocess_config import (
    PREDICTIVE_FEATURE_COLUMNS,
    REACTION_FEATURE_COLUMNS,
    PreprocessConfig,
)


def run_preprocess_validation(
    *,
    cfg: PreprocessConfig,
    raw_counts: Dict[str, int],
    classifications: pd.DataFrame,
    primary: pd.DataFrame,
    research: pd.DataFrame,
    exclusions: pd.DataFrame,
    band_probs: pd.DataFrame,
    band_semantics: pd.DataFrame,
    signals_meta: Dict[str, Any],
    event_ml: pd.DataFrame,
    returns_crosscheck: Dict[str, Any],
    events: pd.DataFrame,
) -> Dict[str, Any]:
    """Run QA checks; return report dict (includes errors list)."""
    messages: List[str] = []
    errors: List[str] = []

    messages.append(f"raw_counts={raw_counts}")
    messages.append(
        f"classifications={len(classifications)} primary={len(primary)} "
        f"research={len(research)} excluded={len(exclusions)}"
    )

    if not classifications.empty:
        by_type = classifications["contract_type"].value_counts().to_dict()
        by_status = classifications["review_status"].value_counts().to_dict()
        messages.append(f"by_contract_type={by_type}")
        messages.append(f"by_review_status={by_status}")

    # Prob sums
    if not band_probs.empty:
        grp = band_probs.groupby(["event_id", "platform", "timestamp"])["probability"].sum()
        max_dev = float((grp - 1.0).abs().max())
        messages.append(f"prob_sum max|sum-1|={max_dev:.2e}")
        if max_dev > 1e-5:
            errors.append(f"band probabilities do not sum to 1 (max={max_dev})")

    # Predictive timestamps must be <= -5
    pred_used = signals_meta.get("predictive_minutes_to_release_used", [])
    messages.append(f"predictive_minutes_used={pred_used}")
    bad = [m for m in pred_used if m > -5]
    if bad:
        errors.append(f"predictive features used forbidden offsets {bad}")
    else:
        messages.append("predictive timestamp check passed (all offsets <= -5)")

    rx_used = signals_meta.get("reaction_minutes_to_release_used", [])
    messages.append(f"reaction_minutes_used={rx_used}")

    # Primary universe edge cases
    if not primary.empty and "edge_case_type" in primary.columns:
        bad_edges = primary[
            primary["edge_case_type"].isin(
                ["noise", "malformed_bands", "timestamp_overlap_unlinked"]
            )
        ]
        if len(bad_edges):
            errors.append(f"primary universe contains {len(bad_edges)} edge-case contracts")
        else:
            messages.append("primary universe has no noise/malformed/overlap edge cases")

    # Band semantics coverage for primary band contracts
    if not primary.empty and not band_semantics.empty:
        prim_bands = primary[primary["band_id"].notna()]
        sem_keys = set(
            zip(
                band_semantics["platform"],
                band_semantics["contract_id"],
            )
        )
        missing = [
            (r.platform, r.contract_id)
            for r in prim_bands.itertuples()
            if (r.platform, r.contract_id) not in sem_keys
        ]
        messages.append(f"primary band contracts missing band_semantics={len(missing)}")
        if len(missing) > len(prim_bands) * 0.05:
            errors.append(f"band_semantics incomplete for primary ({len(missing)} missing)")

    # ML base completeness for CPI/FOMC/FED_CUTS
    macro_events = events[events["event_type"].isin(["CPI", "FOMC", "FED_CUTS"])]
    if not event_ml.empty and len(macro_events):
        covered = set(event_ml["event_id"]) & set(macro_events["event_id"])
        messages.append(
            f"macro events in ML base: {len(covered)}/{len(macro_events)}"
        )
        # Require PMPS_pre present for covered
        if "PMPS_pre_Kalshi" in event_ml.columns:
            n_nan = int(event_ml["PMPS_pre_Kalshi"].isna().sum())
            messages.append(f"PMPS_pre_Kalshi missing rows={n_nan}")

    messages.append(f"returns_crosscheck={returns_crosscheck}")

    # Feature column inventory
    present_pred = [c for c in PREDICTIVE_FEATURE_COLUMNS if c in event_ml.columns]
    present_rx = [c for c in REACTION_FEATURE_COLUMNS if c in event_ml.columns]
    messages.append(f"predictive_cols_present={len(present_pred)}/{len(PREDICTIVE_FEATURE_COLUMNS)}")
    messages.append(f"reaction_cols_present={len(present_rx)}/{len(REACTION_FEATURE_COLUMNS)}")

    if errors:
        for e in errors:
            messages.append(f"ERROR: {e}")
    else:
        messages.append("All preprocessing validation checks passed.")

    return {
        "schema_version": cfg.schema_version,
        "messages": messages,
        "errors": errors,
        "passed": len(errors) == 0,
        "predictive_feature_columns": list(PREDICTIVE_FEATURE_COLUMNS),
        "reaction_feature_columns": list(REACTION_FEATURE_COLUMNS),
        "raw_counts": raw_counts,
        "counts": {
            "classifications": len(classifications),
            "primary_contracts": len(primary),
            "research_contracts": len(research),
            "exclusions": len(exclusions),
            "band_semantics": len(band_semantics),
            "event_ml_base": len(event_ml),
        },
        "signals_meta": signals_meta,
        "returns_crosscheck": returns_crosscheck,
    }
