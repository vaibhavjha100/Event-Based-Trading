"""Optional contract_event_map load or deterministic rebuild."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def build_contract_event_map(
    classifications: pd.DataFrame,
    kalshi: pd.DataFrame,
    polymarket: pd.DataFrame,
    existing_map: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Use existing map if present; otherwise build from metadata + classifications."""
    if existing_map is not None and len(existing_map):
        m = existing_map.copy()
        # Enrich with classification fields
        cls = classifications[
            ["platform", "contract_id", "contract_type", "usable_for_primary", "review_status"]
        ]
        m = m.merge(cls, on=["platform", "contract_id"], how="left")
        m["map_source"] = "raw_contract_event_map"
        return m

    rows = []
    cls_idx = classifications.set_index(["platform", "contract_id"])

    for _, c in kalshi.iterrows():
        key = ("KALSHI", c["ticker"])
        cl = cls_idx.loc[key] if key in cls_idx.index else None
        eid = c["event_id"] if pd.notna(c.get("event_id")) else pd.NA
        edge = str(c.get("edge_case_type", "none"))
        usable = bool(cl["usable_for_primary"]) if cl is not None else edge == "none"
        rows.append(
            {
                "platform": "KALSHI",
                "contract_id": c["ticker"],
                "event_id": eid,
                "macro_variable": pd.NA,
                "band_id": c["band_id"] if pd.notna(c.get("band_id")) else pd.NA,
                "is_core_macro_contract": usable and edge == "none" and pd.notna(eid),
                "edge_case_type": edge,
                "contract_type": cl["contract_type"] if cl is not None else "AMBIGUOUS",
                "usable_for_primary": usable,
                "review_status": cl["review_status"] if cl is not None else "ambiguous",
                "map_source": "rebuilt",
            }
        )

    for _, c in polymarket.iterrows():
        key = ("POLYMARKET", c["condition_id"])
        cl = cls_idx.loc[key] if key in cls_idx.index else None
        eid = c["event_id"] if pd.notna(c.get("event_id")) else pd.NA
        edge = str(c.get("edge_case_type", "none"))
        usable = bool(cl["usable_for_primary"]) if cl is not None else edge == "none"
        rows.append(
            {
                "platform": "POLYMARKET",
                "contract_id": c["condition_id"],
                "event_id": eid,
                "macro_variable": pd.NA,
                "band_id": c["band_id"] if pd.notna(c.get("band_id")) else pd.NA,
                "is_core_macro_contract": usable and edge == "none" and pd.notna(eid),
                "edge_case_type": edge,
                "contract_type": cl["contract_type"] if cl is not None else "AMBIGUOUS",
                "usable_for_primary": usable,
                "review_status": cl["review_status"] if cl is not None else "ambiguous",
                "map_source": "rebuilt",
            }
        )

    return pd.DataFrame(rows)
