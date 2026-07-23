import json
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_loader import (
    REQUIRED_METRIC_KEYS,
    REQUIRED_PERFORMANCE_COLUMNS,
    REQUIRED_SIGNAL_COLUMNS,
    demo_metrics,
    demo_performance,
    demo_signals,
    load_data,
)


class ConstantReturnModel:
    def predict(self, rows: pd.DataFrame) -> list[float]:
        return [0.002 if index % 2 == 0 else -0.002 for index in range(len(rows))]


class DataLoaderTests(unittest.TestCase):
    def test_demo_outputs_match_required_schema(self) -> None:
        self.assertTrue(REQUIRED_SIGNAL_COLUMNS.issubset(demo_signals().columns))
        self.assertTrue(
            REQUIRED_PERFORMANCE_COLUMNS.issubset(demo_performance().columns)
        )
        self.assertTrue(REQUIRED_METRIC_KEYS.issubset(demo_metrics()))

    def test_missing_files_return_labelled_demo_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            signals, performance, metrics, is_demo, issues = load_data(Path(directory))

        self.assertTrue(is_demo)
        self.assertGreaterEqual(len(issues), 3)
        self.assertFalse(signals.empty)
        self.assertFalse(performance.empty)
        self.assertTrue(metrics["model_version"].startswith("demo-"))

    def test_complete_validated_files_disable_demo_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            demo_signals().head(2).to_csv(data_dir / "live_signals.csv", index=False)
            demo_performance().head(2).to_csv(
                data_dir / "cumulative_returns.csv", index=False
            )
            metrics = demo_metrics()
            metrics["model_version"] = "validated-v1"
            metrics["source"] = "validated test fixture"
            (data_dir / "model_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )

            _, _, loaded_metrics, is_demo, issues = load_data(data_dir)

        self.assertFalse(is_demo)
        self.assertEqual(issues, [])
        self.assertEqual(loaded_metrics["model_version"], "validated-v1")

    def test_missing_required_column_falls_back_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            malformed = demo_signals().drop(columns=["known_at"])
            malformed.to_csv(data_dir / "live_signals.csv", index=False)

            signals, _, _, is_demo, issues = load_data(data_dir)

        self.assertTrue(is_demo)
        self.assertIn("known_at", " ".join(issues))
        self.assertIn("known_at", signals.columns)

    def test_model_inference_path_builds_live_signals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            features = demo_signals().head(2).drop(columns=["signal_direction"])
            features["feature_pmps_pre"] = features["pmps_pre"]
            features["feature_p_bad"] = features["p_bad"]
            features.to_csv(data_dir / "simulated_features.csv", index=False)
            (data_dir / "model_metadata.json").write_text(
                json.dumps(
                    {
                        "model_version": "validated-model-v1",
                        "feature_columns": ["feature_pmps_pre", "feature_p_bad"],
                        "prediction_kind": "return_forecast",
                        "long_threshold": 0.001,
                        "short_threshold": -0.001,
                    }
                ),
                encoding="utf-8",
            )
            with (data_dir / "model_bundle.pkl").open("wb") as handle:
                pickle.dump(ConstantReturnModel(), handle)
            demo_performance().head(2).to_csv(
                data_dir / "cumulative_returns.csv", index=False
            )
            metrics = demo_metrics()
            metrics["model_version"] = "validated-model-v1"
            (data_dir / "model_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )

            signals, _, _, is_demo, issues = load_data(data_dir)

        self.assertFalse(is_demo)
        self.assertEqual(issues, [])
        self.assertEqual(signals["signal_direction"].tolist(), ["Long", "Short"])


if __name__ == "__main__":
    unittest.main()
