"""Band semantics: signed distance from consensus and adverse flags."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ml.preprocessing.timestamps import to_utc_timestamp


def latest_prerelease_consensus(
    consensus: pd.DataFrame, event_id: int, release_time
) -> Optional[pd.Series]:
    """Return consensus row with max consensus_time strictly before t0."""
    t0 = to_utc_timestamp(release_time)
    sub = consensus[consensus["event_id"] == event_id].copy()
    if sub.empty:
        return None
    sub = sub[sub["consensus_time"] < t0]
    if sub.empty:
        return None
    return sub.sort_values("consensus_time").iloc[-1]


def build_band_semantics(
    events: pd.DataFrame,
    consensus: pd.DataFrame,
    classifications: pd.DataFrame,
    band_probs: pd.DataFrame,
    primary: pd.DataFrame,
    research: pd.DataFrame,
) -> pd.DataFrame:
    """Build auditable band-level adverse semantics vs latest pre-release consensus."""
    primary_ids = set(zip(primary["platform"], primary["contract_id"]))
    research_ids = set(zip(research["platform"], research["contract_id"]))

    bp_lookup = (
        band_probs.sort_values("minutes_to_release")
        .groupby(["event_id", "platform", "band_id"], as_index=False)
        .agg(
            band_value=("band_value", "first"),
            band_label_bp=("band_label", "first"),
        )
    )

    rows = []
    cand = classifications[
        classifications["event_id"].notna() & classifications["band_id"].notna()
    ].copy()
    events_idx = events.set_index("event_id")

    for _, c in cand.iterrows():
        eid = int(c["event_id"])
        if eid not in events_idx.index:
            continue
        release = events_idx.loc[eid, "release_time"]
        cons_row = latest_prerelease_consensus(consensus, eid, release)
        if cons_row is None:
            continue
        consensus_value = float(cons_row["consensus_value"])

        platform = c["platform"]
        band_id = int(c["band_id"])
        bp = bp_lookup[
            (bp_lookup["event_id"] == eid)
            & (bp_lookup["platform"] == platform)
            & (bp_lookup["band_id"] == band_id)
        ]

        if len(bp):
            band_value = float(bp.iloc[0]["band_value"])
            band_label = str(bp.iloc[0]["band_label_bp"])
            source = "raw_band_probs"
        else:
            band_value = float(c.get("threshold_or_band") or 0.0)
            band_label = str(c.get("band_label") or "")
            source = str(c.get("band_source") or "rules")

        signed_distance = band_value - consensus_value
        is_adverse = signed_distance > 0

        key = (platform, c["contract_id"])
        if key in primary_ids:
            inclusion = "primary"
        elif key in research_ids:
            inclusion = "research"
        else:
            inclusion = "excluded"

        rows.append(
            {
                "event_id": eid,
                "platform": platform,
                "contract_id": c["contract_id"],
                "band_id": band_id,
                "band_label": band_label,
                "band_value": band_value,
                "consensus_value": consensus_value,
                "signed_distance": signed_distance,
                "is_adverse": bool(is_adverse),
                "inclusion_status": inclusion,
                "band_parse_source": source,
                "review_notes": c.get("notes", ""),
            }
        )

    return pd.DataFrame(rows)
