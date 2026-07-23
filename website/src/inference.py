from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_LATEST_FEATURE_COLUMNS = {
    "date",
    "market_title",
    "category",
    "target_asset",
    "p_bad",
    "pmps_pre",
    "volume",
    "open_interest",
    "classification_confidence",
    "known_at",
}
REQUIRED_MODEL_METADATA_KEYS = {
    "feature_columns",
    "prediction_kind",
}


def _read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_model(data_dir: Path) -> tuple[Any | None, str | None]:
    joblib_path = data_dir / "model_bundle.joblib"
    pickle_path = data_dir / "model_bundle.pkl"

    if joblib_path.exists():
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required to load model_bundle.joblib") from exc
        return _unwrap_model(joblib.load(joblib_path)), joblib_path.name

    if pickle_path.exists():
        # Trusted local team artifact only. Do not expose pickle upload in the app.
        with pickle_path.open("rb") as handle:
            return _unwrap_model(pickle.load(handle)), pickle_path.name

    return None, None


def _unwrap_model(bundle: Any) -> Any:
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]
    if isinstance(bundle, dict) and {"pipeline", "feature_names"}.issubset(bundle):
        from ml.backtesting.website_bundle import SklearnReturnModel

        return SklearnReturnModel(bundle["pipeline"], list(bundle["feature_names"]))
    return bundle


def _feature_path(data_dir: Path) -> Path:
    simulated_path = data_dir / "simulated_features.csv"
    if simulated_path.exists():
        return simulated_path
    return data_dir / "latest_features.csv"


def _thresholds(metadata: dict[str, Any]) -> tuple[float, float, list[str]]:
    if "long_threshold" in metadata and "short_threshold" in metadata:
        return float(metadata["long_threshold"]), float(metadata["short_threshold"]), []

    thresholds = metadata.get("thresholds", {})
    if isinstance(thresholds, dict) and "tau" in thresholds:
        tau = float(thresholds["tau"])
        return tau, -tau, []

    return 0.0, 0.0, [
        "model_metadata.json needs long_threshold/short_threshold or thresholds.tau"
    ]


def _metadata_issues(metadata: dict[str, Any] | None) -> list[str]:
    if metadata is None:
        return ["model_metadata.json is missing"]

    issues: list[str] = []
    missing_metadata = REQUIRED_MODEL_METADATA_KEYS - set(metadata)
    if missing_metadata:
        issues.append(
            "model_metadata.json is missing: "
            + ", ".join(sorted(missing_metadata))
        )

    prediction_kind = metadata.get("prediction_kind")
    if prediction_kind not in {"return_forecast", "continuous_return"}:
        issues.append(
            "model_metadata.json prediction_kind must be return_forecast or continuous_return"
        )

    _, _, threshold_issues = _thresholds(metadata)
    issues.extend(threshold_issues)
    return issues


def infer_live_signals(data_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Build the displayed signal table from a frozen model and feature panel."""
    issues: list[str] = []
    features_path = _feature_path(data_dir)
    metadata_path = data_dir / "model_metadata.json"

    if not features_path.exists() and not metadata_path.exists():
        return (
            None,
            [
                "model inference inputs are missing: simulated_features.csv or latest_features.csv, "
                "model_metadata.json, and model_bundle.joblib or model_bundle.pkl"
            ],
        )

    features = pd.read_csv(features_path) if features_path.exists() else None
    metadata = _read_metadata(metadata_path)

    if features is None:
        issues.append("simulated_features.csv or latest_features.csv is missing")
    issues.extend(_metadata_issues(metadata))

    model = None
    model_name = None
    try:
        model, model_name = _load_model(data_dir)
    except Exception as exc:  # pragma: no cover - exact loader errors are environment-specific
        issues.append(f"model bundle could not be loaded: {exc}")

    if model is None:
        issues.append("model_bundle.joblib or model_bundle.pkl is missing")

    if features is not None:
        missing_feature_columns = REQUIRED_LATEST_FEATURE_COLUMNS - set(features.columns)
        if missing_feature_columns:
            issues.append(
                f"{features_path.name} is missing: "
                + ", ".join(sorted(missing_feature_columns))
            )

    if issues:
        return None, issues

    assert metadata is not None
    assert features is not None
    assert model is not None

    feature_columns = metadata["feature_columns"]
    if not isinstance(feature_columns, list) or not feature_columns:
        return None, ["model_metadata.json feature_columns must be a non-empty list"]

    missing_model_features = set(feature_columns) - set(features.columns)
    if missing_model_features:
        return (
            None,
            [
                f"{features_path.name} is missing model features: "
                + ", ".join(sorted(missing_model_features))
            ],
        )

    try:
        predictions = model.predict(features[feature_columns])
    except Exception as exc:
        return None, [f"model prediction failed: {exc}"]

    output = features[list(REQUIRED_LATEST_FEATURE_COLUMNS)].copy()
    output["model_forecast"] = pd.Series(predictions, index=output.index).astype(float)
    long_threshold, short_threshold, threshold_issues = _thresholds(metadata)
    if threshold_issues:
        return None, threshold_issues
    output["signal_direction"] = "Neutral"
    output.loc[output["model_forecast"] > long_threshold, "signal_direction"] = "Long"
    output.loc[output["model_forecast"] < short_threshold, "signal_direction"] = "Short"
    output["model_artifact"] = model_name
    return output, []
