"""Explicit contract ↔ event mapping table."""

from __future__ import annotations

import pandas as pd


def build_contract_event_map(
    events: pd.DataFrame,
    kalshi: pd.DataFrame,
    polymarket: pd.DataFrame,
) -> pd.DataFrame:
    """Build contract_event_map linking platform contracts to events.

    Core macro contracts have ``is_core_macro_contract=True`` and a non-null
    ``event_id``. Edge cases are labeled via ``edge_case_type``.
    """
    macro_by_event = events.set_index("event_id")["macro_variable"].to_dict()
    rows = []

    for _, c in kalshi.iterrows():
        edge = c["edge_case_type"]
        eid = c["event_id"] if pd.notna(c["event_id"]) else None
        is_core = edge == "none" and eid is not None
        rows.append(
            {
                "platform": "KALSHI",
                "contract_id": c["ticker"],
                "event_id": int(eid) if eid is not None else pd.NA,
                "macro_variable": macro_by_event.get(int(eid)) if eid is not None else pd.NA,
                "band_id": int(c["band_id"]) if pd.notna(c["band_id"]) else pd.NA,
                "is_core_macro_contract": bool(is_core),
                "edge_case_type": edge,
            }
        )

    for _, c in polymarket.iterrows():
        edge = c["edge_case_type"]
        eid = c["event_id"] if pd.notna(c["event_id"]) else None
        # Multi-outcome parent markets (band_id NA, edge none) are also core
        is_core = edge == "none" and eid is not None
        rows.append(
            {
                "platform": "POLYMARKET",
                "contract_id": c["condition_id"],
                "event_id": int(eid) if eid is not None else pd.NA,
                "macro_variable": macro_by_event.get(int(eid)) if eid is not None else pd.NA,
                "band_id": int(c["band_id"]) if pd.notna(c["band_id"]) else pd.NA,
                "is_core_macro_contract": bool(is_core),
                "edge_case_type": edge,
            }
        )

    return pd.DataFrame(rows)
