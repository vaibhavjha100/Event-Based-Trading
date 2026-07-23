from __future__ import annotations

import numpy as np
import pandas as pd


class SklearnReturnModel:
    """Compatibility wrapper for Data Lead model bundles."""

    def __init__(self, pipeline, feature_columns: list[str]):
        self.pipeline = pipeline
        self.feature_columns = list(feature_columns)

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [column for column in self.feature_columns if column not in X.columns]
            if missing:
                raise KeyError(f"Missing feature columns: {missing}")
            frame = X[self.feature_columns]
        else:
            matrix = np.asarray(X, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            if matrix.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"Expected {len(self.feature_columns)} columns, got {matrix.shape[1]}"
                )
            frame = pd.DataFrame(matrix, columns=self.feature_columns)
        return np.asarray(self.pipeline.predict(frame), dtype=float)
