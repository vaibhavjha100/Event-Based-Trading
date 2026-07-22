"""Primary / research universe selection and exclusions audit."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from ml.preprocess_config import PreprocessConfig


def select_universes(
    classifications: pd.DataFrame,
    contract_map: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return primary universe, research universe, and exclusions audit."""
    merged = classifications.merge(
        contract_map[
            [
                "platform",
                "contract_id",
                "event_id",
                "band_id",
                "is_core_macro_contract",
                "edge_case_type",
            ]
        ],
        on=["platform", "contract_id"],
        how="left",
        suffixes=("", "_map"),
    )
    # Prefer classification event_id; fill from map
    if "event_id_map" in merged.columns:
        merged["event_id"] = merged["event_id"].fillna(merged["event_id_map"])

    exclusions = []
    primary_mask = []
    research_mask = []

    for i, r in merged.iterrows():
        reasons = []
        edge = str(r.get("edge_case_type", "none"))
        ctype = str(r.get("contract_type", "AMBIGUOUS"))
        conf = float(r.get("confidence", 0) or 0)
        review = str(r.get("review_status", "ambiguous"))

        if edge == "noise":
            reasons.append("edge_noise")
        if edge == "malformed_bands" or bool(r.get("malformed", False)):
            reasons.append("malformed")
        if edge == "timestamp_overlap_unlinked":
            reasons.append("timestamp_overlap_unlinked")
        if ctype == "NON_MACRO":
            reasons.append("non_macro")
        if ctype == "AMBIGUOUS":
            reasons.append("ambiguous_type")
        if conf < cfg.confidence_min_primary and ctype in cfg.primary_types:
            reasons.append("low_confidence")
        if review == "excluded":
            reasons.append("review_excluded")
        if review == "ambiguous":
            reasons.append("review_ambiguous")

        in_primary = (
            ctype in cfg.primary_types
            and bool(r.get("usable_for_primary", False))
            and not bool(r.get("malformed", False))
            and edge == "none"
            and review == "retained"
            and conf >= cfg.confidence_min_primary
        )
        in_research = in_primary or (
            ctype in cfg.research_extra_types
            and edge == "none"
            and not bool(r.get("malformed", False))
            and review == "retained"
        )

        primary_mask.append(in_primary)
        research_mask.append(in_research)
        if not in_research:
            exclusions.append(
                {
                    "platform": r["platform"],
                    "contract_id": r["contract_id"],
                    "contract_type": ctype,
                    "edge_case_type": edge,
                    "review_status": review,
                    "reason_codes": "|".join(reasons) if reasons else "not_in_research",
                    "classifier_source": r.get("family_source", "rules"),
                }
            )

    merged["in_primary"] = primary_mask
    merged["in_research"] = research_mask

    primary = merged[merged["in_primary"]].copy()
    research = merged[merged["in_research"]].copy()
    excl = pd.DataFrame(exclusions)
    return primary, research, excl
