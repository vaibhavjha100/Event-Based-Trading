"""Gemini family classification + contract-level band extraction with cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml.preprocess_config import PreprocessConfig

FAMILY_SCHEMA = {
    "type": "object",
    "properties": {
        "contract_type": {
            "type": "string",
            "enum": ["CPI", "FOMC", "FED_CUTS", "OTHER_MACRO", "NON_MACRO", "AMBIGUOUS"],
        },
        "macro_topic": {"type": "string"},
        "inferred_event_date": {"type": "string"},
        "contract_family": {"type": "string"},
        "usable_for_primary": {"type": "boolean"},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
        "malformed": {"type": "boolean"},
    },
    "required": [
        "contract_type",
        "macro_topic",
        "contract_family",
        "usable_for_primary",
        "confidence",
        "notes",
        "malformed",
    ],
}

BAND_SCHEMA = {
    "type": "object",
    "properties": {
        "band_label": {"type": "string"},
        "threshold_or_band": {"type": "number"},
        "band_lower": {"type": ["number", "null"]},
        "band_upper": {"type": ["number", "null"]},
        "directionality": {"type": "string"},
        "signed_distance_hint": {"type": ["number", "null"]},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["band_label", "threshold_or_band", "directionality", "confidence"],
}


def _cache_key(tier: str, platform: str, entity_id: str, raw_text: str) -> str:
    h = hashlib.sha256(f"{tier}|{platform}|{entity_id}|{raw_text}".encode()).hexdigest()
    return h


class GeminiCache:
    """JSONL cache for Gemini (and rule-based) extraction results."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, dict] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self._mem[obj["cache_key"]] = obj
                    except json.JSONDecodeError:
                        continue

    def get(self, key: str) -> Optional[dict]:
        return self._mem.get(key)

    def put(self, key: str, payload: dict) -> None:
        payload = {**payload, "cache_key": key}
        self._mem[key] = payload
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")


def _load_api_key() -> Optional[str]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")


_GEMINI_DISABLED_SESSION = False


def _gemini_generate(model: str, prompt: str, schema: dict) -> dict:
    global _GEMINI_DISABLED_SESSION
    if _GEMINI_DISABLED_SESSION:
        raise RuntimeError("gemini_disabled_session_quota")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_load_api_key())
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            _GEMINI_DISABLED_SESSION = True
        raise
    text = response.text or "{}"
    return json.loads(text)


def rule_family_classify(
    platform: str, family_id: str, title: str, edge_case_type: str
) -> dict:
    """Deterministic type classification from ticker/title patterns."""
    text = f"{family_id} {title}".upper()
    edge = (edge_case_type or "none").lower()

    if edge in ("noise",) or "NOISE" in text:
        return _family_result("NON_MACRO", False, 0.95, "rule:noise", True if False else False)
    if edge == "malformed_bands" or "MALFORM" in text:
        return _family_result("AMBIGUOUS", False, 0.9, "rule:malformed", True)
    if edge == "timestamp_overlap_unlinked" or "OVERLAP" in text:
        return _family_result("NON_MACRO", False, 0.9, "rule:overlap_unlinked", False)

    if "FED_CUT" in text or "FED CUT" in text or "RATE CUT" in text:
        return _family_result("FED_CUTS", True, 0.9, "rule:fed_cuts", False)
    if "FOMC" in text or "POLICYRATE" in text or "POLICY RATE" in text:
        return _family_result("FOMC", True, 0.9, "rule:fomc", False)
    if "CPI" in text:
        return _family_result("CPI", True, 0.95, "rule:cpi", False)
    if "NFP" in text or "UNEMPLOYMENT" in text:
        return _family_result("OTHER_MACRO", False, 0.85, "rule:nfp", False)
    if "GAS" in text or "TSA" in text or "AI " in text or "CRYPTO" in text:
        return _family_result("NON_MACRO", False, 0.9, "rule:non_macro_topic", False)

    return _family_result("AMBIGUOUS", False, 0.4, "rule:unknown", False)


def _family_result(
    contract_type: str,
    usable: bool,
    conf: float,
    notes: str,
    malformed: bool,
) -> dict:
    return {
        "contract_type": contract_type,
        "macro_topic": contract_type,
        "inferred_event_date": "",
        "contract_family": notes,
        "usable_for_primary": usable and contract_type in ("CPI", "FOMC", "FED_CUTS"),
        "confidence": conf,
        "notes": notes,
        "malformed": malformed,
        "source": "rules",
    }


def parse_band_label(label: str, consensus: Optional[float] = None) -> dict:
    """Parse band label text into threshold / directionality fields."""
    label = str(label) if label is not None else ""
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", label.replace(",", ""))]
    direction = "unknown"
    low = up = None
    threshold = None

    lower_l = label.lower()
    if "above" in lower_l:
        direction = "above"
        threshold = nums[0] if nums else None
        low = threshold
    elif "below" in lower_l:
        direction = "below"
        threshold = nums[0] if nums else None
        up = threshold
    elif " to " in lower_l and len(nums) >= 2:
        direction = "range"
        low, up = nums[0], nums[1]
        threshold = 0.5 * (low + up)
    elif len(nums) == 1:
        threshold = nums[0]
        direction = "point"
    elif len(nums) >= 2:
        low, up = nums[0], nums[1]
        threshold = 0.5 * (low + up)
        direction = "range"

    if threshold is None:
        threshold = 0.0

    hint = None
    if consensus is not None and threshold is not None:
        hint = float(threshold) - float(consensus)

    return {
        "band_label": label,
        "threshold_or_band": float(threshold),
        "band_lower": low,
        "band_upper": up,
        "directionality": direction,
        "signed_distance_hint": hint,
        "confidence": 0.85 if nums else 0.3,
        "notes": "rule:band_parse",
        "source": "rules",
    }


def classify_contracts(
    kalshi: pd.DataFrame,
    polymarket: pd.DataFrame,
    cfg: PreprocessConfig,
) -> pd.DataFrame:
    """Run family-level type classification and contract-level band extraction."""
    cache = GeminiCache(cfg.cache_path())
    use_gemini = cfg.gemini_enabled and bool(_load_api_key())
    rows: List[dict] = []

    # --- Kalshi families ---
    for family_id, g in kalshi.groupby("event_ticker"):
        sample = g.iloc[0]
        title = str(sample.get("title", ""))
        edge = str(sample.get("edge_case_type", "none"))
        raw_text = f"{family_id}|{title}|bands=" + ";".join(
            str(x) for x in g["yes_sub_title"].head(8).tolist()
        )
        fam = _classify_family(
            cache,
            cfg,
            use_gemini,
            platform="KALSHI",
            entity_id=str(family_id),
            raw_text=raw_text,
            title=title,
            edge=edge,
        )
        for _, c in g.iterrows():
            band = _extract_band(
                cache,
                cfg,
                use_gemini,
                platform="KALSHI",
                contract_id=str(c["ticker"]),
                label=str(c.get("yes_sub_title", "")),
                title=str(c.get("title", "")),
            )
            rows.append(_classification_row("KALSHI", c["ticker"], family_id, fam, band, c))

    # --- Polymarket: family classify by event_id when present, else per slug ---
    poly = polymarket.copy()
    poly["_family_key"] = poly.apply(
        lambda r: f"event:{int(r['event_id'])}"
        if pd.notna(r.get("event_id"))
        else f"slug:{r.get('slug', r['condition_id'])}",
        axis=1,
    )

    family_results: Dict[str, dict] = {}
    for family_key, g in poly.groupby("_family_key"):
        sample = g.iloc[0]
        title = str(sample.get("question", ""))
        edge = str(sample.get("edge_case_type", "none"))
        family_id = str(sample.get("slug", sample["condition_id"]))
        raw_fam = f"{family_key}|{family_id}|{title}|edge={edge}"
        family_results[family_key] = _classify_family(
            cache,
            cfg,
            use_gemini,
            platform="POLYMARKET",
            entity_id=str(family_key),
            raw_text=raw_fam,
            title=title,
            edge=edge,
        )

    for _, c in poly.iterrows():
        family_key = c["_family_key"]
        fam = family_results[family_key]
        family_id = str(c.get("slug", c["condition_id"]))
        title = str(c.get("question", ""))
        label = title.split("—")[-1].strip() if "—" in title else str(c.get("outcomes", ""))
        band = _extract_band(
            cache,
            cfg,
            use_gemini,
            platform="POLYMARKET",
            contract_id=str(c["condition_id"]),
            label=label,
            title=title,
        )
        rows.append(
            _classification_row("POLYMARKET", c["condition_id"], family_id, fam, band, c)
        )

    return pd.DataFrame(rows)


def _classify_family(
    cache: GeminiCache,
    cfg: PreprocessConfig,
    use_gemini: bool,
    *,
    platform: str,
    entity_id: str,
    raw_text: str,
    title: str,
    edge: str,
) -> dict:
    key = _cache_key("family", platform, entity_id, raw_text)
    cached = cache.get(key)
    if cached and not cfg.gemini_force:
        if cfg.gemini_failed_only and cached.get("status") != "error":
            return cached["result"]
        if not cfg.gemini_failed_only:
            return cached["result"]

    result = None
    status = "ok"
    source = "rules"
    if use_gemini:
        try:
            prompt = (
                "Classify this prediction-market contract family for macro event research.\n"
                f"Platform: {platform}\nFamily id: {entity_id}\nText:\n{raw_text}\n"
                "Return structured fields only."
            )
            result = _gemini_generate(cfg.gemini_model, prompt, FAMILY_SCHEMA)
            result["source"] = "gemini"
            source = "gemini"
        except Exception as exc:  # noqa: BLE001
            status = "error"
            result = rule_family_classify(platform, entity_id, title, edge)
            result["notes"] = f"gemini_error:{exc}; fallback={result['notes']}"
            result["source"] = "rules"
    else:
        result = rule_family_classify(platform, entity_id, title, edge)

    cache.put(
        key,
        {
            "tier": "family",
            "platform": platform,
            "entity_id": entity_id,
            "raw_text": raw_text,
            "status": status,
            "source": source,
            "result": result,
        },
    )
    return result


def _extract_band(
    cache: GeminiCache,
    cfg: PreprocessConfig,
    use_gemini: bool,
    *,
    platform: str,
    contract_id: str,
    label: str,
    title: str,
) -> dict:
    raw_text = f"{title}|{label}"
    key = _cache_key("band", platform, contract_id, raw_text)
    cached = cache.get(key)
    if cached and not cfg.gemini_force:
        if cfg.gemini_failed_only and cached.get("status") != "error":
            return cached["result"]
        if not cfg.gemini_failed_only:
            return cached["result"]

    # Prefer deterministic parse; optionally refine with Gemini
    parsed = parse_band_label(label)
    status = "ok"
    source = "rules"

    if use_gemini and parsed["confidence"] < 0.6:
        try:
            prompt = (
                "Extract band/threshold semantics from this prediction-market contract.\n"
                f"Platform: {platform}\nContract: {contract_id}\n"
                f"Title: {title}\nOutcome/band text: {label}\n"
                "threshold_or_band should be the representative numeric level."
            )
            result = _gemini_generate(cfg.gemini_model, prompt, BAND_SCHEMA)
            result["source"] = "gemini"
            result.setdefault("band_lower", None)
            result.setdefault("band_upper", None)
            result.setdefault("signed_distance_hint", None)
            result.setdefault("notes", "")
            parsed = result
            source = "gemini"
        except Exception as exc:  # noqa: BLE001
            status = "error"
            parsed["notes"] = f"gemini_error:{exc}; {parsed['notes']}"

    cache.put(
        key,
        {
            "tier": "band",
            "platform": platform,
            "entity_id": contract_id,
            "raw_text": raw_text,
            "status": status,
            "source": source,
            "result": parsed,
        },
    )
    return parsed


def _classification_row(
    platform: str,
    contract_id: str,
    family_id: str,
    fam: dict,
    band: dict,
    contract_row: pd.Series,
) -> dict:
    edge = str(contract_row.get("edge_case_type", "none"))
    ctype = fam["contract_type"]
    malformed = bool(fam.get("malformed", False)) or edge == "malformed_bands"
    usable = bool(fam.get("usable_for_primary", False)) and not malformed
    conf = float(fam.get("confidence", 0) or 0)

    if (
        malformed
        or ctype == "NON_MACRO"
        or edge in ("noise", "timestamp_overlap_unlinked", "malformed_bands")
    ):
        review = "excluded"
    elif ctype == "AMBIGUOUS" or (ctype in ("CPI", "FOMC", "FED_CUTS") and conf < 0.45):
        review = "ambiguous"
    elif ctype in ("CPI", "FOMC", "FED_CUTS") and usable:
        review = "retained"
    elif ctype == "OTHER_MACRO" and edge == "none" and not malformed:
        review = "retained"
    else:
        review = "excluded"

    return {
        "platform": platform,
        "contract_id": contract_id,
        "family_id": family_id,
        "raw_title": contract_row.get("title") or contract_row.get("question"),
        "raw_outcome_label": contract_row.get("yes_sub_title")
        or contract_row.get("outcomes"),
        "event_id": contract_row.get("event_id")
        if pd.notna(contract_row.get("event_id"))
        else pd.NA,
        "band_id": contract_row.get("band_id")
        if pd.notna(contract_row.get("band_id"))
        else pd.NA,
        "edge_case_type": edge,
        "contract_type": ctype,
        "macro_topic": fam.get("macro_topic"),
        "inferred_event_date": fam.get("inferred_event_date", ""),
        "contract_family": fam.get("contract_family"),
        "usable_for_primary": usable and ctype in ("CPI", "FOMC", "FED_CUTS"),
        "confidence": fam.get("confidence"),
        "malformed": malformed,
        "notes": fam.get("notes"),
        "band_label": band.get("band_label"),
        "threshold_or_band": band.get("threshold_or_band"),
        "band_lower": band.get("band_lower"),
        "band_upper": band.get("band_upper"),
        "directionality": band.get("directionality"),
        "band_source": band.get("source", "rules"),
        "family_source": fam.get("source", "rules"),
        "review_status": review,
    }
